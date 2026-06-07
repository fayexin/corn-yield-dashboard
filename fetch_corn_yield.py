"""
Fetch county-level corn and soybean yield (bu/acre) from USDA NASS Quick Stats.

Pulls all CONUS counties, 1980 to present, one crop-year per request to stay
under the 50,000-record API cap, then writes a tidy CSV for the dashboard.

Setup:
    1. Get a free API key: https://quickstats.nass.usda.gov/api
    2. export NASS_API_KEY="your-key-here"
    3. python fetch_corn_yield.py

Output:
    data/crop_yield_county_1980_present.csv
    columns: crop, fips, state_alpha, county_name, year, yield_bu_acre
"""

import os
import time
from datetime import datetime

import pandas as pd
import requests


API_URL = "https://quickstats.nass.usda.gov/api/api_GET/"
API_KEY = os.environ.get("NASS_API_KEY", "")

START_YEAR = 1980
END_YEAR = datetime.now().year

OUTPUT_PATH = "data/crop_yield_county_1980_present.csv"

# Alaska (02) and Hawaii (15) sit outside the continental US.
NON_CONUS_STATE_FIPS = {"02", "15"}

# Each crop maps to its NASS commodity and the exact bu/acre yield series.
CROPS = {
    "corn": {
        "commodity_desc": "CORN",
        "short_desc": "CORN, GRAIN - YIELD, MEASURED IN BU / ACRE",
    },
    "soybeans": {
        "commodity_desc": "SOYBEANS",
        "short_desc": "SOYBEANS - YIELD, MEASURED IN BU / ACRE",
    },
}

BASE_PARAMS = {
    "key": API_KEY,
    "source_desc": "SURVEY",
    "statisticcat_desc": "YIELD",
    "agg_level_desc": "COUNTY",
    "format": "JSON",
}


def fetch_crop_year(crop_config, year):
    params = dict(BASE_PARAMS, year=year, **crop_config)
    response = requests.get(API_URL, params=params, timeout=60)

    # NASS returns a non-200 or an "error" payload when no records match.
    if response.status_code != 200:
        return pd.DataFrame()

    payload = response.json()

    if "data" not in payload:
        return pd.DataFrame()

    return pd.DataFrame(payload["data"])


def main():
    if not API_KEY:
        raise SystemExit(
            "Set NASS_API_KEY first. Get a free key at "
            "https://quickstats.nass.usda.gov/api"
        )

    frames = []

    for crop, crop_config in CROPS.items():
        for year in range(START_YEAR, END_YEAR + 1):
            df_year = fetch_crop_year(crop_config, year)

            if df_year.empty:
                print(f"{crop} {year}: no records")
                continue

            df_year["crop"] = crop
            frames.append(df_year)
            print(f"{crop} {year}: {len(df_year)} county records")
            time.sleep(1)  # be polite to the API

    raw = pd.concat(frames, ignore_index=True)

    # Drop combined/district aggregates, which carry no county_ansi.
    raw = raw[raw["county_ansi"].astype(str).str.strip() != ""]
    raw = raw[~raw["state_ansi"].isin(NON_CONUS_STATE_FIPS)]

    # Build the 5-digit FIPS used by the dashboard's county GeoJSON.
    raw["fips"] = (
        raw["state_ansi"].astype(str).str.zfill(2)
        + raw["county_ansi"].astype(str).str.zfill(3)
    )

    # Coerce yield to numeric; suppressed values like "(D)" become NaN and drop.
    raw["yield_bu_acre"] = pd.to_numeric(
        raw["Value"].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )
    raw = raw.dropna(subset=["yield_bu_acre"])

    tidy = raw[
        ["crop", "fips", "state_alpha", "county_name", "year", "yield_bu_acre"]
    ].copy()

    tidy["year"] = tidy["year"].astype(int)
    tidy = tidy.sort_values(["crop", "fips", "year"]).reset_index(drop=True)

    tidy.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved {len(tidy):,} rows to {OUTPUT_PATH}")
    for crop in CROPS:
        subset = tidy[tidy["crop"] == crop]
        print(
            f"  {crop}: {len(subset):,} rows, "
            f"{subset['fips'].nunique():,} counties, "
            f"{subset['year'].min()}-{subset['year'].max()}"
        )


if __name__ == "__main__":
    main()