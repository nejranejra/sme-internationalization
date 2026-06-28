"""
02_clean.py
Build a clean firm-year panel from the most recent raw pull.

Run:  python code/02_clean.py
Input:  newest folder in data/raw/  (the timestamped folder from 01_pull_data.py)
Output: data/processed/panel_clean.parquet
        data/processed/clean_log.txt
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROC_DIR = PROJECT_ROOT / "data" / "processed"

NUMERIC_COLS = [
    "fyear", "ib", "at", "sale", "seq", "xrd",
    "dltt", "capx", "che", "emp",  
]


def newest_raw_folder() -> Path:
    folders = [p for p in RAW_DIR.iterdir() if p.is_dir()]
    if not folders:
        raise SystemExit(
            "No raw data folder found in data/raw/. "
            "Run 'python code/01_pull_data.py' first."
        )
    return max(folders, key=lambda p: p.stat().st_mtime)


def main() -> None:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    src = newest_raw_folder()
    print(f"Reading raw chunks from {src}")

    chunks = sorted(src.glob("fyear_*.parquet"))
    if not chunks:
        raise SystemExit(f"No fyear_*.parquet files in {src}")

    df = pd.concat((pd.read_parquet(f) for f in chunks), ignore_index=True)
    raw_rows = len(df)
    print(f"  combined raw rows: {raw_rows:,}")

    # Drop rows missing the panel keys
    df = df.dropna(subset=["gvkey", "fyear"])

    # Convert object columns to numeric where expected
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Drop exact duplicates and duplicate firm-year keys (keep first)
    df = df.drop_duplicates()
    df = df.drop_duplicates(subset=["gvkey", "fyear"], keep="first")

    # Sort for a tidy panel
    df = df.sort_values(["gvkey", "fyear"]).reset_index(drop=True)

    clean_rows = len(df)
    out = PROC_DIR / "panel_clean.parquet"
    df.to_parquet(out, index=False)

    log = [
        f"raw rows: {raw_rows}",
        f"clean rows: {clean_rows}",
        f"columns: {df.shape[1]}",
        f"firms (gvkey): {df['gvkey'].nunique()}",
        f"years: {int(df['fyear'].min())}-{int(df['fyear'].max())}",
    ]
    (PROC_DIR / "clean_log.txt").write_text("\n".join(log))

    print("\n".join(log))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
