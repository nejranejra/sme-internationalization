"""
04_regression.py
Estimate the R&D intensity -> RoA relationship with three models:
    (1) Pooled OLS
    (2) Two-way fixed effects (firm + year)   [linearmodels PanelOLS]
    (3) TWFE with the R&D x Size interaction   [H2]

Also prints a Hausman-style OLS vs FE comparison and H1/H2 diagnostics.

Run:  python code/04_regression.py
Output: output/tables/regression_results.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.panel import PanelOLS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = PROJECT_ROOT / "data" / "processed"
TAB_DIR = PROJECT_ROOT / "output" / "tables"

DV = "roa"
X_MAIN = "rd_intensity"
INTERACT = "rd_x_size"          # X * moderator
CONTROLS = ["ln_at", "leverage", "capx_intensity", "cash_ratio"]


def get_se(res, name):
    """Return standard error for a parameter from either a statsmodels or a
    linearmodels result object (their APIs differ)."""
    try:
        return res.std_errors[name]      # linearmodels
    except (AttributeError, KeyError, TypeError):
        pass
    try:
        return res.bse[name]             # statsmodels
    except (AttributeError, KeyError):
        return np.nan


def get_param(res, name):
    try:
        return res.params[name]
    except (KeyError, TypeError):
        return np.nan


def get_p(res, name):
    try:
        return res.pvalues[name]
    except (KeyError, TypeError):
        return np.nan


def stars(p):
    if p != p:  # NaN
        return ""
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""


def main() -> None:
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PROC_DIR / "panel_with_vars.parquet")

    # Panel index for linearmodels: (entity, time)
    # Force all model variables to numeric (WRDS Global sometimes returns text)
    for _c in [DV, X_MAIN, INTERACT] + CONTROLS:
        if _c in df.columns:
            df[_c] = pd.to_numeric(df[_c], errors="coerce")
    df = df.dropna(subset=[DV, X_MAIN] + CONTROLS).copy()
    df["fyear"] = df["fyear"].astype(int)
    # COERCE_ALL_NUMERIC
    _model_cols = [DV, X_MAIN, INTERACT] + CONTROLS + ['gvkey', 'fyear']
    for _c in _model_cols:
        if _c in df.columns and _c != 'gvkey':
            df[_c] = pd.to_numeric(df[_c], errors='coerce')
    # Cast nullable Float64 -> plain float64 so statsmodels accepts it
    _num = [DV, X_MAIN, INTERACT] + CONTROLS
    for _c in _num:
        if _c in df.columns:
            df[_c] = df[_c].astype('float64')
    df = df.dropna(subset=_num).copy()
    panel = df.set_index(["gvkey", "fyear"])

    # ---- Model (1): Pooled OLS ---------------------------------------------
    X1 = sm.add_constant(df[[X_MAIN] + CONTROLS])
    m1 = sm.OLS(df[DV], X1).fit(cov_type="HC1")

    # ---- Model (2): Two-way fixed effects ----------------------------------
    m2 = PanelOLS(
        panel[DV], panel[[X_MAIN] + CONTROLS],
        entity_effects=True, time_effects=True,
    ).fit(cov_type="clustered", cluster_entity=True)

    # ---- Model (3): TWFE + interaction (H2) --------------------------------
    m3 = PanelOLS(
        panel[DV], panel[[X_MAIN, INTERACT] + CONTROLS],
        entity_effects=True, time_effects=True,
    ).fit(cov_type="clustered", cluster_entity=True)

    print("Model (1) pooled OLS  R2 =", round(m1.rsquared, 4))
    print("Model (2) TWFE        R2 =", round(m2.rsquared, 4))
    print("Model (3) TWFE+inter  R2 =", round(m3.rsquared, 4))

    # ---- Build results table ------------------------------------------------
    rows = []
    terms = [X_MAIN, INTERACT] + CONTROLS + ["const"]
    models = [("(1) OLS", m1), ("(2) TWFE", m2), ("(3) TWFE+int", m3)]
    for term in terms:
        row = {"variable": term}
        for label, res in models:
            b = get_param(res, term)
            se = get_se(res, term)
            p = get_p(res, term)
            if b == b:  # not NaN
                row[label] = f"{b:.4f}{stars(p)}"
                row[f"{label} se"] = f"({se:.4f})" if se == se else ""
            else:
                row[label] = ""
                row[f"{label} se"] = ""
        rows.append(row)

    # R-squared and N rows
    r2_row = {"variable": "R-squared"}
    n_row = {"variable": "N"}
    for label, res in models:
        r2_row[label] = f"{res.rsquared:.4f}"
        r2_row[f"{label} se"] = ""
        n = int(res.nobs)
        n_row[label] = f"{n}"
        n_row[f"{label} se"] = ""
    rows += [r2_row, n_row]

    table = pd.DataFrame(rows)
    table.to_csv(TAB_DIR / "regression_results.csv", index=False)
    print(f"Saved {TAB_DIR / 'regression_results.csv'}")

    # ---- OLS vs FE comparison (omitted variable bias) -----------------------
    b_ols = get_param(m1, X_MAIN)
    b_fe = get_param(m2, X_MAIN)
    if b_ols not in (0, np.nan) and b_ols == b_ols:
        diff_pct = 100 * (b_fe - b_ols) / abs(b_ols)
        print(f"\nOLS vs FE on {X_MAIN}: OLS={b_ols:.4f}, FE={b_fe:.4f} "
              f"({diff_pct:+.1f}% change). Large change => omitted variable bias in OLS.")

    # ---- H1 diagnostic ------------------------------------------------------
    b1, p1 = get_param(m2, X_MAIN), get_p(m2, X_MAIN)
    print("\n--- H1: R&D intensity -> RoA ---")
    if b1 < 0 and p1 < 0.05:
        print(f"H1 SUPPORTED: beta={b1:.4f} (p={p1:.3f}), negative and significant.")
        print("Interpretation: R&D is expensed, reducing current earnings (RoA).")
    elif b1 > 0 and p1 < 0.05:
        print(f"H1 (opposite sign): beta={b1:.4f} (p={p1:.3f}), positive and significant.")
    else:
        print(f"H1 NOT SUPPORTED: beta={b1:.4f} (p={p1:.3f}), not significant at 5%.")

    # ---- H2 diagnostic ------------------------------------------------------
    b2, p2 = get_param(m3, INTERACT), get_p(m3, INTERACT)
    print("\n--- H2: firm size moderates R&D -> RoA ---")
    if p2 < 0.05:
        direction = "strengthens" if b2 > 0 else "weakens"
        print(f"H2 SUPPORTED: interaction beta={b2:.4f} (p={p2:.3f}); "
              f"larger size {direction} the R&D-RoA relationship.")
    else:
        print(f"H2 NOT SUPPORTED: interaction beta={b2:.4f} (p={p2:.3f}), "
              f"not significant at 5%.")


if __name__ == "__main__":
    main()
