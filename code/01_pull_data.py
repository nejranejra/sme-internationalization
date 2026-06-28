"""
01_pull_data.py
Pull Compustat Global Fundamentals Annual from WRDS.

Pulls ALL firms for fiscal years 2015-2024 and saves one parquet file per
year into a NEW timestamped folder under data/raw/.

Run:  python code/01_pull_data.py
Needs: a .env file in the project root containing WRDS_USERNAME=your_username
"""

import os
import datetime as dt
from pathlib import Path

import pandas as pd
import wrds
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
START_YEAR = 2015
END_YEAR = 2024
TABLE = "comp_global_daily.g_funda"

# Every field used downstream (Y, X, interaction, controls) must be pulled.
# Pulling a generous superset is fine and matches the "pull everything" design.
VARIABLES = [
    "gvkey", "fyear", "datadate", "conm", "fic", "curcd",
    "ib", "at", "sale", "seq", "xrd", "dltt", "capx", "che", "emp",
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    username = os.getenv("WRDS_USERNAME")
    if not username:
        raise SystemExit(
            "WRDS_USERNAME not found. Create a .env file in the project root "
            "with the line:  WRDS_USERNAME=your_wrds_username"
        )

    print(f"Connecting to WRDS as {username} ...")
    db = wrds.Connection(wrds_username=username)

    stamp = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = RAW_DIR / stamp
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving into {out_dir}")

    cols = ", ".join(VARIABLES)
    total_rows = 0
    meta_lines = [f"Pull timestamp: {stamp}", f"Table: {TABLE}", ""]

    for year in range(START_YEAR, END_YEAR + 1):
        query = (
            f"SELECT {cols} FROM {TABLE} "
            f"WHERE fyear = {year} "
            f""
        )
        print(f"  pulling fyear {year} ...", end=" ", flush=True)
        try:
            df = db.raw_sql(query)
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED ({exc})")
            continue
        df.to_parquet(out_dir / f"fyear_{year}.parquet", index=False)
        print(f"{len(df):>8,} rows")
        total_rows += len(df)
        meta_lines.append(f"fyear_{year}: {len(df)} rows")

    db.close()

    # Save schema + metadata so provenance is documented (S5 requirement)
    schema = pd.DataFrame({"variable": VARIABLES})
    schema.to_csv(out_dir / "column_schema.csv", index=False)
    meta_lines += ["", f"Total rows: {total_rows}"]
    (out_dir / "pull_metadata.txt").write_text("\n".join(meta_lines))

    print(f"\nDone. Total rows pulled: {total_rows:,}")
    print(f"Raw data in: {out_dir}")


if __name__ == "__main__":
    main()
