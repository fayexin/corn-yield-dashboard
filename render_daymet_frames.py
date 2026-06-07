from pathlib import Path
from urllib.request import urlretrieve

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import Normalize
from tqdm import tqdm


YEAR = 2025
VARIABLE = "tmax"

DATA_DIR = Path("data/daymet_monthly")
GEO_DIR = Path("data/geo")
OUTPUT_DIR = Path(f"data/daymet_frames/{VARIABLE}/{YEAR}")

GEO_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

COUNTY_GEOJSON = GEO_DIR / "geojson-counties-fips.json"
STATE_GEOJSON = GEO_DIR / "us-states.json"

COUNTY_URL = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
STATE_URL = "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"

VARIABLE_LABELS = {
    "tmax": "Maximum temperature",
    "tmin": "Minimum temperature",
    "tmean": "Mean temperature",
    "gdd10": "Growing degree days, base 10°C",
    "prcp": "Precipitation",
    "srad": "Shortwave radiation",
    "vp": "Water vapor pressure",
    "vpd": "Vapor pressure deficit",
    "swe": "Snow water equivalent",
    "dayl": "Day length",
}

VARIABLE_UNITS = {
    "tmax": "°C",
    "tmin": "°C",
    "tmean": "°C",
    "gdd10": "°C day",
    "prcp": "mm/day",
    "srad": "W/m²",
    "vp": "Pa",
    "vpd": "kPa",
    "swe": "kg/m²",
    "dayl": "seconds",
}

VARIABLE_CMAPS = {
    "tmax": "YlOrRd",
    "tmin": "Blues",
    "tmean": "RdYlBu_r",
    "gdd10": "YlOrRd",
    "prcp": "Blues",
    "srad": "YlOrBr",
    "vp": "PuBuGn",
    "vpd": "plasma",
    "swe": "Blues",
    "dayl": "viridis",
}

STATE_NAME_TO_ABBR = {
    "Alabama": "AL",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "District of Columbia": "DC",
    "Florida": "FL",
    "Georgia": "GA",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
}


def download_geojson_files():
    if not COUNTY_GEOJSON.exists():
        print("Downloading county GeoJSON...")
        urlretrieve(COUNTY_URL, COUNTY_GEOJSON)

    if not STATE_GEOJSON.exists():
        print("Downloading state GeoJSON...")
        urlretrieve(STATE_URL, STATE_GEOJSON)


def load_year_data():
    files = sorted(DATA_DIR.glob(f"daymet_{YEAR}_*.parquet"))

    if not files:
        raise FileNotFoundError(f"No monthly Parquet files found for {YEAR}.")

    data_frames = []

    for file in files:
        print(f"Reading {file}")
        month_df = pd.read_parquet(
            file,
            columns=["fips", "state", "date", VARIABLE],
        )
        data_frames.append(month_df)

    df = pd.concat(data_frames, ignore_index=True)
    df["fips"] = df["fips"].astype(str).str.zfill(5)
    df["date"] = pd.to_datetime(df["date"])

    return df


def load_geometries(available_fips, available_states):
    counties = gpd.read_file(COUNTY_GEOJSON)
    counties["fips"] = counties["id"].astype(str).str.zfill(5)
    counties = counties[counties["fips"].isin(available_fips)].copy()

    states = gpd.read_file(STATE_GEOJSON)
    states["state"] = states["name"].map(STATE_NAME_TO_ABBR)
    states = states[states["state"].isin(available_states)].copy()

    counties = counties.to_crs("EPSG:5070")
    states = states.to_crs("EPSG:5070")

    return counties, states


def add_state_labels(ax, states):
    label_points = states.copy()
    label_points["geometry"] = label_points.geometry.representative_point()

    for _, row in label_points.iterrows():
        ax.text(
            row.geometry.x,
            row.geometry.y,
            row["state"],
            ha="center",
            va="center",
            fontsize=7,
            color="#555555",
            alpha=0.45,
        )


def render_frame(day_df, counties, states, selected_date, vmin, vmax):
    output_file = OUTPUT_DIR / f"{selected_date}.webp"

    if output_file.exists():
        return

    merged = counties.merge(
        day_df[["fips", VARIABLE]],
        on="fips",
        how="left",
    )

    fig, ax = plt.subplots(figsize=(18, 10.5))

    merged.plot(
        column=VARIABLE,
        ax=ax,
        cmap=VARIABLE_CMAPS[VARIABLE],
        vmin=vmin,
        vmax=vmax,
        linewidth=0,
        missing_kwds={"color": "#eeeeee"},
    )

    states.boundary.plot(
        ax=ax,
        linewidth=0.8,
        color="#444444",
        alpha=0.75,
    )

    add_state_labels(ax, states)

    ax.set_axis_off()

    minx, miny, maxx, maxy = counties.total_bounds
    x_margin = (maxx - minx) * 0.03
    y_margin = (maxy - miny) * 0.04
    ax.set_xlim(minx - x_margin, maxx + x_margin)
    ax.set_ylim(miny - y_margin, maxy + y_margin)

    title = f"{VARIABLE_LABELS[VARIABLE]} ({VARIABLE}), {selected_date}"
    ax.set_title(title, fontsize=17, pad=12)

    norm = Normalize(vmin=vmin, vmax=vmax)
    scalar_map = plt.cm.ScalarMappable(
        norm=norm,
        cmap=VARIABLE_CMAPS[VARIABLE],
    )
    scalar_map.set_array([])

    colorbar = fig.colorbar(
        scalar_map,
        ax=ax,
        fraction=0.030,
        pad=0.015,
    )
    colorbar.set_label(
        f"{VARIABLE_LABELS[VARIABLE]} ({VARIABLE_UNITS[VARIABLE]})",
        fontsize=10,
    )
    colorbar.ax.tick_params(labelsize=9)

    fig.savefig(
        output_file,
        format="webp",
        dpi=200,
        bbox_inches="tight",
        pad_inches=0.05,
        facecolor="white",
        pil_kwargs={"quality": 95},
    )

    plt.close(fig)


def main():
    download_geojson_files()

    df = load_year_data()

    available_fips = set(df["fips"].unique())
    available_states = set(df["state"].unique())

    counties, states = load_geometries(available_fips, available_states)

    vmin = df[VARIABLE].quantile(0.01)
    vmax = df[VARIABLE].quantile(0.99)

    print(f"Using fixed color range for {VARIABLE}: {vmin:.2f} to {vmax:.2f}")

    date_values = sorted(df["date"].dt.date.unique())

    for selected_date in tqdm(date_values):
        day_df = df[df["date"].dt.date == selected_date].copy()
        render_frame(
            day_df=day_df,
            counties=counties,
            states=states,
            selected_date=selected_date,
            vmin=vmin,
            vmax=vmax,
        )

    print(f"Done. Frames saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()