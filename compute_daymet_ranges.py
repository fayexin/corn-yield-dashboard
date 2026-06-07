from pathlib import Path

import pandas as pd


DATA_DIR = Path("data/daymet_monthly")
OUTPUT_FILE = Path("data/daymet_variable_ranges.csv")

VARIABLES = [
    "tmax",
    "tmin",
    "tmean",
    "gdd10",
    "prcp",
    "srad",
    "vp",
    "vpd",
    "swe",
    "dayl",
]


records = []

for variable in VARIABLES:
    print(f"Processing {variable}...")

    values = []

    for file in sorted(DATA_DIR.glob("daymet_*.parquet")):
        df = pd.read_parquet(file, columns=[variable])
        values.append(df[variable])

    all_values = pd.concat(values, ignore_index=True)

    records.append(
        {
            "variable": variable,
            "min": all_values.min(),
            "max": all_values.max(),
            "p01": all_values.quantile(0.01),
            "p99": all_values.quantile(0.99),
        }
    )

ranges = pd.DataFrame(records)
ranges.to_csv(OUTPUT_FILE, index=False)

print(f"Saved {OUTPUT_FILE}")
print(ranges)