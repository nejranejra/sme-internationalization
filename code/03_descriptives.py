"""
03_descriptives.py
Build variables, descriptive statistics, and exploratory figures.

Research design (Session 6): R&D intensity -> firm performance (RoA),
moderated by firm size.
    Y  (DV)        : roa            = ib / at
    X  (IV)        : rd_intensity   = xrd.fillna(0) / at
    interaction    : rd_x_size      = rd_intensity * ln_at
    controls       : ln_at, leverage, capx_intensity, cash_ratio

Run:  python code/03_descriptives.py
Output:
    output/tables/summary_statistics.csv
    output/figures/correlation_matrix.png
    output/figures/dv_distribution.png
    output/figures/main_relationship.png
    data/processed/panel_with_vars.parquet
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = PROJECT_ROOT / "data" / "processed"
TAB_DIR = PROJECT_ROOT / "output" / "tables"
FIG_DIR = PROJECT_ROOT / "output" / "figures"

CORE_VARS = ["roa", "rd_intensity", "ln_at", "leverage"]
WINSORIZE_VARS = ["roa", "rd_intensity", "leverage", "capx_intensity", "cash_ratio"]
VAR_LABELS = {
    "roa": "RoA (ib/at)",
    "rd_intensity": "R&D intensity (xrd/at)",
    "rd_x_size": "R&D x Size",
    "ln_at": "Firm size ln(at)",
    "leverage": "Leverage (dltt/at)",
    "capx_intensity": "CAPX intensity (capx/at)",
    "cash_ratio": "Cash ratio (che/at)",
}


def winsorize(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    lo, hi = s.quantile(lower), s.quantile(upper)
    return s.clip(lo, hi)


def main() -> None:
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(PROC_DIR / "panel_clean.parquet")
    print(f"Loaded panel: {df.shape[0]:,} rows")

    # -- Data quality filters -------------------------------------------------
    df = df[(df["at"] > 0.1) & (df["sale"] > 0) & (df["seq"] > 0)].copy()
    # Remove micro-firms (negative log assets distort size plots)
    df = df[df["at"] >= 1].copy()  # keep firms with >= EUR 1m assets
    # EU SME filter (emp in thousands; at in millions)
    sme_mask = (df["emp"] < 0.25) | (df["at"] <= 43)
    df = df[sme_mask].copy()
    print(f"After filters: {df.shape[0]:,} rows")

    # -- Variable construction ------------------------------------------------
    df["roa"] = df["ib"] / df["at"]                       # DV (Y)
    df["rd_intensity"] = df["xrd"].fillna(0) / df["at"]   # IV (X)
    df["ln_at"] = np.log(df["at"])                        # moderator + control
    df["rd_x_size"] = df["rd_intensity"] * df["ln_at"]    # interaction (H2)
    df["leverage"] = df["dltt"] / df["at"]                # control
    df["capx_intensity"] = df["capx"] / df["at"]          # control
    df["cash_ratio"] = df["che"] / df["at"]               # control

    # Drop rows where core vars are missing
    df = df.dropna(subset=CORE_VARS).copy()
    n_rd = int((df["rd_intensity"] > 0).sum())
    print(f"Firms reporting R&D (rd_intensity > 0): {n_rd:,}")

    # -- Winsorize (NOT ln_at: already bounded) -------------------------------
    print("Winsorize ranges (1%-99%):")
    for v in WINSORIZE_VARS:
        if v in df.columns:
            lo, hi = df[v].quantile(0.01), df[v].quantile(0.99)
            df[v] = winsorize(df[v])
            print(f"  {v:<16} [{lo:.4f}, {hi:.4f}]")

    # -- Summary statistics ---------------------------------------------------
    stat_vars = [v for v in VAR_LABELS if v in df.columns]
    summary = df[stat_vars].describe().T[
        ["count", "mean", "std", "min", "50%", "max"]
    ]
    summary = summary.rename(columns={"50%": "median"})
    summary.index = [VAR_LABELS[v] for v in stat_vars]
    summary.round(4).to_csv(TAB_DIR / "summary_statistics.csv")
    print(f"Saved {TAB_DIR / 'summary_statistics.csv'}")

    # -- Correlation matrix figure -------------------------------------------
    corr_vars = ["roa", "rd_intensity", "ln_at", "leverage",
                 "capx_intensity", "cash_ratio"]
    corr_vars = [v for v in corr_vars if v in df.columns]
    corr = df[corr_vars].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                vmin=-1, vmax=1, square=True,
                xticklabels=[VAR_LABELS[v] for v in corr_vars],
                yticklabels=[VAR_LABELS[v] for v in corr_vars])
    plt.title("Correlation matrix")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "correlation_matrix.png", dpi=150)
    plt.close()

    # -- DV distribution figure ----------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].hist(df["roa"], bins=60, color="#1f3b57")
    ax[0].set_title("Distribution of RoA")
    ax[0].set_xlabel(VAR_LABELS["roa"])
    ax[1].hist(df["ln_at"], bins=60, color="#1f3b57")
    ax[1].set_title("Distribution of firm size")
    ax[1].set_xlabel(VAR_LABELS["ln_at"])
    plt.tight_layout()
    plt.savefig(FIG_DIR / "dv_distribution.png", dpi=150)
    plt.close()

    # -- Main relationship figure --------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    # Left: scatter + linear fit on R&D-reporting firms
    has_rd = df[df["rd_intensity"] > 0]
    ax[0].scatter(has_rd["rd_intensity"], has_rd["roa"],
                  s=6, alpha=0.3, color="#c0392b")
    if len(has_rd) > 2:
        b, a = np.polyfit(has_rd["rd_intensity"], has_rd["roa"], 1)
        xs = np.linspace(has_rd["rd_intensity"].min(),
                         has_rd["rd_intensity"].max(), 50)
        ax[0].plot(xs, a + b * xs, color="black", lw=2)
    ax[0].set_xlabel(VAR_LABELS["rd_intensity"])
    ax[0].set_ylabel(VAR_LABELS["roa"])
    ax[0].set_title("R&D intensity vs RoA")
    # Right: RoA by R&D status
    no_rd = df[df["rd_intensity"] == 0]
    ax[1].scatter(no_rd["ln_at"], no_rd["roa"], s=6, alpha=0.3,
                  color="#2980b9", label="No R&D")
    ax[1].scatter(has_rd["ln_at"], has_rd["roa"], s=6, alpha=0.3,
                  color="#c0392b", label="Has R&D")
    ax[1].set_xlabel(VAR_LABELS["ln_at"])
    ax[1].set_ylabel(VAR_LABELS["roa"])
    ax[1].set_title("RoA by firm size and R&D status")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "main_relationship.png", dpi=150)
    plt.close()

    # -- Save panel with constructed variables --------------------------------
    out = PROC_DIR / "panel_with_vars.parquet"
    df.to_parquet(out, index=False)
    print(f"Saved {out}")
    print("Descriptives complete.")


if __name__ == "__main__":
    main()
