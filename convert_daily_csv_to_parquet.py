"""
Convert raw daily Daymet CSVs (1980-2025) into monthly Parquet files with the
same schema as data/daymet_monthly/daymet_YYYY_MM.parquet.

Reads each large CSV in chunks (bounded memory), splits rows by month, and
streams them into one Parquet file per month. Run locally; the output is
several GB and must NOT be committed to the repo. Its purpose is to feed
build_monthly_means.py (extending the monthly timeline to 1980) and any
local daily browsing.

Usage:
    1. Put the CSV paths in INPUT_FILES below (or leave the glob).
    2. If your CSV column names differ from the expected ones, fill in
       COLUMN_MAP, e.g. {"FIPS": "fips", "tmax_degC": "tmax"}.
    3. python convert_daily_csv_to_parquet.py
    4. Then build the full monthly means:
       set DAYMET_MONTHLY_DIR=<OUTPUT_DIR>   (Windows cmd)
       python build_monthly_means.py

Output:
    <OUTPUT_DIR>/daymet_YYYY_MM.parquet
"""

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


# ---------------------------------------------------------------- config ---

INPUT_FILES = sorted(glob.glob("E:/coding/daymet_raw/daymet_conus_county_daily_*.csv"))

# Output also stays on the local drive, outside the repo entirely.
OUTPUT_DIR = Path("E:/coding/daymet_monthly_full")

# Map your CSV's column names to the expected ones (left: CSV, right: target).
COLUMN_MAP = {"FIPS": "fips"}

CHUNK_ROWS = 1_000_000

# A monthly parquet that already has county names, used to fill namelsad/state
# /county if the CSVs lack them.
NAME_SOURCE = Path("data/daymet_monthly/daymet_2025_01.parquet")

VARIABLES = [
    "tmax", "tmin", "tmean", "gdd10", "prcp",
    "srad", "vp", "vpd", "swe", "dayl",
]
META = ["fips", "state", "county", "namelsad"]

# --------------------------------------------------------------- helpers ---


def load_name_lookup():
    if not NAME_SOURCE.exists():
        return None

    ref = pd.read_parquet(NAME_SOURCE, columns=META)
    ref["fips"] = ref["fips"].astype(str).str.zfill(5)

    return ref.drop_duplicates("fips").set_index("fips")


def derive_missing(df):
    if "tmean" not in df.columns and {"tmax", "tmin"} <= set(df.columns):
        df["tmean"] = (df["tmax"] + df["tmin"]) / 2.0

    if "gdd10" not in df.columns and "tmean" in df.columns:
        df["gdd10"] = np.clip(df["tmean"] - 10.0, 0.0, None)

    if "vpd" not in df.columns and {"vp", "tmean"} <= set(df.columns):
        # Magnus saturation vapor pressure (Pa), then deficit in kPa.
        saturation = 610.7 * np.exp(
            17.38 * df["tmean"] / (239.0 + df["tmean"])
        )
        df["vpd"] = np.clip((saturation - df["vp"]) / 1000.0, 0.0, None)

    return df


def validate(df, source):
    missing = [c for c in ["fips", "date"] if c not in df.columns]
    missing_vars = [v for v in VARIABLES if v not in df.columns]

    if missing or missing_vars:
        raise SystemExit(
            f"{source}: missing required columns {missing + missing_vars}.\n"
            f"Columns found: {list(df.columns)}\n"
            "Fill in COLUMN_MAP at the top of this script to rename your "
            "CSV columns to the expected names."
        )


def process_file(path, names):
    writers = {}
    schema = None

    reader = pd.read_csv(path, chunksize=CHUNK_ROWS)

    for chunk_number, chunk in enumerate(reader):
        if COLUMN_MAP:
            chunk = chunk.rename(columns=COLUMN_MAP)

        if chunk_number == 0:
            base_missing = [c for c in ("fips", "date") if c not in chunk.columns]

            if base_missing:
                raise SystemExit(
                    f"{path}: required column(s) {base_missing} not found.\n"
                    f"Columns in this CSV: {list(chunk.columns)}\n"
                    "Fill in COLUMN_MAP at the top of this script to map your "
                    "column names to the expected ones."
                )

        chunk["fips"] = chunk["fips"].astype(str).str.zfill(5)
        chunk["date"] = pd.to_datetime(chunk["date"])
        chunk["year"] = chunk["date"].dt.year.astype("int32")
        chunk["month"] = chunk["date"].dt.month.astype("int32")

        chunk = derive_missing(chunk)

        if names is not None:
            for column in ["state", "county", "namelsad"]:
                if column not in chunk.columns:
                    chunk[column] = chunk["fips"].map(names[column])

        if chunk_number == 0:
            validate(chunk, path)

        for variable in VARIABLES:
            chunk[variable] = chunk[variable].astype("float32")

        columns = META + ["date", "year", "month", "doy"] + VARIABLES
        chunk = chunk[[c for c in columns if c in chunk.columns]]

        for (year, month), group in chunk.groupby(["year", "month"]):
            key = (int(year), int(month))

            if key not in writers:
                out_path = OUTPUT_DIR / f"daymet_{year}_{month:02d}.parquet"
                table = pa.Table.from_pandas(group, preserve_index=False)
                schema = table.schema
                writers[key] = pq.ParquetWriter(out_path, schema)
                writers[key].write_table(table)
            else:
                table = pa.Table.from_pandas(
                    group, preserve_index=False
                ).cast(schema)
                writers[key].write_table(table)

        print(f"{Path(path).name}: chunk {chunk_number + 1} done")

    for writer in writers.values():
        writer.close()

    print(f"{Path(path).name}: wrote {len(writers)} monthly files")


def main():
    if not INPUT_FILES:
        raise SystemExit(
            "No input CSVs found. Set INPUT_FILES at the top of this script."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    names = load_name_lookup()

    if names is None:
        print(
            "Note: name lookup parquet not found; namelsad/state/county must "
            "exist in the CSVs themselves."
        )

    for path in INPUT_FILES:
        process_file(path, names)

    files = sorted(OUTPUT_DIR.glob("daymet_*.parquet"))
    print(f"\nDone: {len(files)} monthly files in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()