"""
Aggregate daily Daymet monthly Parquet files into compact monthly means.

For each county and month, this averages the daily values of every variable.
The result is small enough to ship in the repo even across many years, and it
powers the interactive monthly timeline page.

Re-run this whenever new daymet_YYYY_MM.parquet files are added to
data/daymet_monthly (for example, after processing the raw 1980-2020 record
into the same schema).

Usage:
    python build_monthly_means.py

Output:
    data/daymet_monthly_means.parquet
    columns: fips, state, county, namelsad, year, month + variable means
"""

from pathlib import Path
import os

import pandas as pd


# Defaults to the repo's recent-years folder. Point it at a local folder
# holding the full 1980-present monthly Parquet to build the complete record:
#   DAYMET_MONTHLY_DIR=/path/to/all_years python build_monthly_means.py
DATA_DIR = Path(os.environ.get("DAYMET_MONTHLY_DIR", "data/daymet_monthly"))
OUTPUT_PATH = Path("data/daymet_monthly_means.parquet")

VARIABLES = [
    "tmax", "tmin", "tmean", "gdd10", "prcp",
    "srad", "vp", "vpd", "swe", "dayl",
]
META = ["fips", "state", "county", "namelsad"]


def main():
    files = sorted(DATA_DIR.glob("daymet_*.parquet"))

    if not files:
        raise SystemExit(f"No Parquet files found in {DATA_DIR}.")

    monthly = []

    for file in files:
        df = pd.read_parquet(file, columns=META + ["year", "month"] + VARIABLES)
        df["fips"] = df["fips"].astype(str).str.zfill(5)

        grouped = df.groupby(
            META + ["year", "month"], as_index=False
        )[VARIABLES].mean()

        monthly.append(grouped)
        print(f"{file.name}: {len(grouped):,} county rows")

    means = pd.concat(monthly, ignore_index=True)

    for variable in VARIABLES:
        means[variable] = means[variable].astype("float32")

    means = means.sort_values(["fips", "year", "month"]).reset_index(drop=True)
    means.to_parquet(OUTPUT_PATH, index=False)

    print(f"\nSaved {len(means):,} rows to {OUTPUT_PATH}")
    print(
        f"Range: {means['year'].min()}-{means['year'].max()}, "
        f"counties: {means['fips'].nunique():,}"
    )


if __name__ == "__main__":
    main()
