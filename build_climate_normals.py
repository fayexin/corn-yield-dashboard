"""
Build county-level climate normals from the monthly-means file.

For each county and calendar month, this computes the 1980-2010 average of
selected variables. The Extreme Weather Explorer page measures every month
against these normals to map anomalies (departures from normal).

Run after data/daymet_monthly_means.parquet covers the full record:
    python build_climate_normals.py

Output:
    data/daymet_normals.parquet
    columns: fips, month, tmean_normal, tmax_normal, prcp_normal
"""

from pathlib import Path

import pandas as pd


MEANS_PATH = Path("data/daymet_monthly_means.parquet")
OUTPUT_PATH = Path("data/daymet_normals.parquet")

BASELINE_START = 1980
BASELINE_END = 2010

VARIABLES = ["tmean", "tmax", "prcp"]


def main():
    means = pd.read_parquet(
        MEANS_PATH, columns=["fips", "year", "month"] + VARIABLES
    )
    means["fips"] = means["fips"].astype(str).str.zfill(5)

    baseline = means[
        (means["year"] >= BASELINE_START) & (means["year"] <= BASELINE_END)
    ]

    if baseline.empty:
        baseline = means
        print(
            f"WARNING: no data in {BASELINE_START}-{BASELINE_END}; using the "
            f"full available range {means['year'].min()}-{means['year'].max()} "
            "as a stand-in baseline. Rebuild once the full record is present."
        )

    normals = baseline.groupby(["fips", "month"], as_index=False)[
        VARIABLES
    ].mean()

    normals = normals.rename(
        columns={variable: f"{variable}_normal" for variable in VARIABLES}
    )

    for column in normals.columns:
        if column.endswith("_normal"):
            normals[column] = normals[column].astype("float32")

    normals.to_parquet(OUTPUT_PATH, index=False)

    print(
        f"Saved {len(normals):,} county-month normals "
        f"({baseline['year'].min()}-{baseline['year'].max()} baseline) "
        f"to {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
