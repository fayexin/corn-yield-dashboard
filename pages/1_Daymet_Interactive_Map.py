import json
from calendar import month_name
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
import plotly.colors as pc
import pydeck as pdk
import streamlit as st

import pydeck.bindings.json_tools as _pdk_json

# pydeck pretty-prints its spec (indent=2), which more than triples the
# payload sent to the browser on every rerun. Serialize compactly instead.
_pdk_json.serialize = lambda obj: json.dumps(
    obj,
    sort_keys=True,
    default=_pdk_json.default_serialize,
    separators=(",", ":"),
)


st.set_page_config(
    page_title="Daymet Interactive Map",
    layout="wide"
)


DATA_DIR = Path("data/daymet_monthly")

BACKGROUND_RGBA = [236, 236, 236, 255]

# Counties whose FIPS codes changed; the geojson uses the old code while the
# data uses the new one. Display the new-code data on the old-code geometry.
#   46113 Shannon County, SD -> 46102 Oglala Lakota County, SD (renamed 2015)
#   51515 Bedford City, VA   -> 51019 Bedford County, VA (merged 2013)
FIPS_ALIASES = {"46113": "46102", "51515": "51019"}


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

VARIABLE_COLOR_SCALES = {
    "tmax": "YlOrRd",
    "tmin": "Blues",
    "tmean": "RdYlBu_r",
    "gdd10": "YlOrRd",
    "prcp": "Blues",
    "srad": "YlOrBr",
    "vp": "PuBuGn",
    "vpd": "Plasma",
    "swe": "Blues",
    "dayl": "Viridis",
}


@st.cache_data
def list_available_files():
    files = sorted(DATA_DIR.glob("daymet_*.parquet"))

    records = []

    for file in files:
        parts = file.stem.split("_")

        if len(parts) == 3:
            records.append(
                {
                    "year": int(parts[1]),
                    "month": int(parts[2]),
                    "file": str(file),
                }
            )

    return pd.DataFrame(records)


@st.cache_data(max_entries=3, show_spinner="Loading selected month...")
def load_month_data(file_path):
    df = pd.read_parquet(file_path)

    df["fips"] = df["fips"].astype(str).str.zfill(5)
    df["date"] = pd.to_datetime(df["date"])

    if "namelsad" not in df.columns:
        df["namelsad"] = df["county"].astype(str) + " County"

    return df


@st.cache_data
def load_variable_ranges():
    ranges = pd.read_csv("data/daymet_variable_ranges.csv")
    return ranges.set_index("variable").to_dict(orient="index")


def _round_coords(value):
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, list):
        return [_round_coords(item) for item in value]
    return value


@st.cache_data(show_spinner="Loading county geometry...")
def load_county_geometries():
    url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"

    with urlopen(url) as response:
        geojson = json.load(response)

    return [
        {
            "fips": feature["id"],
            "geometry": {
                "type": feature["geometry"]["type"],
                "coordinates": _round_coords(feature["geometry"]["coordinates"]),
            },
        }
        for feature in geojson["features"]
    ]


STATE_NAME_TO_ABBR = {
    "Alabama": "AL", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA",
    "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
    "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME",
    "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI",
    "Minnesota": "MN", "Mississippi": "MS", "Missouri": "MO", "Montana": "MT",
    "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ",
    "New Mexico": "NM", "New York": "NY", "North Carolina": "NC",
    "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR",
    "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}


@st.cache_data(show_spinner="Loading state boundaries...")
def load_state_layers():
    url = "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"

    with urlopen(url) as response:
        states = json.load(response)

    border_features = []
    label_points = []

    for feature in states["features"]:
        state_name = feature["properties"]["name"]
        abbr = STATE_NAME_TO_ABBR.get(state_name)

        if abbr is None:
            continue

        geometry = feature["geometry"]

        if geometry["type"] == "Polygon":
            polygons = [geometry["coordinates"]]
        elif geometry["type"] == "MultiPolygon":
            polygons = geometry["coordinates"]
        else:
            continue

        border_features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": geometry["type"],
                    "coordinates": _round_coords(geometry["coordinates"]),
                },
                "properties": {},
            }
        )

        all_lons = []
        all_lats = []

        for polygon in polygons:
            outer_ring = polygon[0]
            all_lons.extend(point[0] for point in outer_ring)
            all_lats.extend(point[1] for point in outer_ring)

        if all_lons:
            label_points.append(
                {
                    "position": [
                        round(sum(all_lons) / len(all_lons), 3),
                        round(sum(all_lats) / len(all_lats), 3),
                    ],
                    "text": abbr,
                }
            )

    return border_features, label_points


@st.cache_data
def get_color_lut(scale_name):
    samples = pc.sample_colorscale(scale_name, np.linspace(0, 1, 256))

    lut = []
    for color in samples:
        if color.startswith("#"):
            lut.append(
                [int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)]
            )
        else:
            nums = color[color.index("(") + 1 : color.index(")")].split(",")
            lut.append([int(round(float(n))) for n in nums[:3]])

    return lut


def build_features(geometries, values, names, lut, vmin, vmax, unit):
    span = max(vmax - vmin, 1e-9)
    features = []

    for county in geometries:
        fips = county["fips"]
        value = values.get(fips)

        if value is None or pd.isna(value):
            fill = BACKGROUND_RGBA
            value_text = "No data"
        else:
            index = int(np.clip((value - vmin) / span, 0.0, 1.0) * 255)
            fill = lut[index] + [255]
            value_text = f"{value:.2f} {unit}"

        features.append(
            {
                "type": "Feature",
                "geometry": county["geometry"],
                "properties": {
                    "name": names.get(fips, f"County {fips}"),
                    "value": value_text,
                    "fill": fill,
                },
            }
        )

    return features


def nice_ticks(vmin, vmax, target_count=10):
    span = max(vmax - vmin, 1e-9)
    raw_step = span / target_count

    magnitude = 10 ** np.floor(np.log10(raw_step))
    for multiple in (1, 2, 2.5, 5, 10):
        step = multiple * magnitude
        if step >= raw_step:
            break

    first = np.ceil(vmin / step) * step
    ticks = [
        0.0 if abs(tick) < step * 1e-6 else float(tick)
        for tick in np.arange(first, vmax + step * 1e-6, step)
    ]

    decimals = 0 if step >= 1 else 1
    return ticks, decimals


def legend_html(lut, vmin, vmax, label, unit):
    bar_height = 480
    span = max(vmax - vmin, 1e-9)

    stops = ", ".join(
        f"rgb({c[0]},{c[1]},{c[2]})" for c in lut[:: max(len(lut) // 24, 1)]
    )

    ticks, decimals = nice_ticks(vmin, vmax)

    tick_items = []
    for tick in ticks:
        bottom = (tick - vmin) / span * bar_height
        text = f"{tick:.{decimals}f}"
        tick_items.append(
            f'<div style="position:absolute; bottom:{bottom - 8:.0f}px; left:0;">'
            f'<span style="display:inline-block; width:8px; height:1px;'
            f' background:#888; vertical-align:middle;"></span>'
            f'<span style="font-size:12px; color:#444; margin-left:3px;'
            f' vertical-align:middle;">{text}</span></div>'
        )

    return f"""
    <div style="display:flex; align-items:center; height:{bar_height + 40}px;">
      <div style="
          width:18px; height:{bar_height}px; border:1px solid #bbb; border-radius:3px;
          background:linear-gradient(to top, {stops});"></div>
      <div style="position:relative; height:{bar_height}px; width:60px;
                  margin-left:2px;">
        {''.join(tick_items)}
      </div>
    </div>
    <div style="font-size:12px; color:#444; margin-top:4px; max-width:90px;">
      {label} ({unit})
    </div>
    """


st.title("Daily Daymet Interactive Map")

st.write(
    "This page visualizes county-level daily Daymet variables across the continental "
    "United States. Select a year, month, date, and variable to view daily spatial "
    "patterns. Pan and zoom the map freely, and hover over a county to read its value."
)


file_index = list_available_files()

if file_index.empty:
    st.error(
        "No Parquet files were found. Please check that files exist in data/daymet_monthly."
    )
    st.stop()


geometries = load_county_geometries()
state_borders, state_labels = load_state_layers()
variable_ranges = load_variable_ranges()


st.sidebar.header("Controls")

available_years = sorted(file_index["year"].unique())

selected_year = st.sidebar.select_slider(
    "Year",
    options=available_years,
    value=available_years[-1],
)

available_months = sorted(
    file_index.loc[file_index["year"] == selected_year, "month"].unique()
)

selected_month = st.sidebar.select_slider(
    "Month",
    options=available_months,
    value=available_months[0],
    format_func=lambda month: month_name[month][:3],
)

selected_file = file_index.loc[
    (file_index["year"] == selected_year)
    & (file_index["month"] == selected_month),
    "file",
].iloc[0]

df = load_month_data(selected_file)

selected_variable = st.sidebar.selectbox(
    "Daymet variable",
    list(VARIABLE_LABELS),
    format_func=lambda variable: f"{VARIABLE_LABELS[variable]} ({variable})",
)

date_options = sorted(df["date"].dt.date.unique())

range_info = variable_ranges[selected_variable]
vmin, vmax = float(range_info["p01"]), float(range_info["p99"])

unit = VARIABLE_UNITS[selected_variable]
label = VARIABLE_LABELS[selected_variable]
lut = get_color_lut(VARIABLE_COLOR_SCALES[selected_variable])


def make_deck(day_date):
    filtered = df[df["date"].dt.date == day_date]

    values = dict(zip(filtered["fips"], filtered[selected_variable]))
    names = dict(
        zip(filtered["fips"], filtered["namelsad"] + ", " + filtered["state"])
    )

    for old_fips, new_fips in FIPS_ALIASES.items():
        if new_fips in values:
            values[old_fips] = values[new_fips]
            names[old_fips] = names[new_fips]

    features = build_features(geometries, values, names, lut, vmin, vmax, unit)

    county_layer = pdk.Layer(
        "GeoJsonLayer",
        data={"type": "FeatureCollection", "features": features},
        get_fill_color="properties.fill",
        get_line_color=[90, 90, 90, 60],
        line_width_min_pixels=0.4,
        pickable=True,
        stroked=True,
        filled=True,
    )

    border_layer = pdk.Layer(
        "GeoJsonLayer",
        data={"type": "FeatureCollection", "features": state_borders},
        get_line_color=[40, 40, 40, 190],
        line_width_min_pixels=1.2,
        stroked=True,
        filled=False,
        pickable=False,
    )

    label_layer = pdk.Layer(
        "TextLayer",
        data=state_labels,
        get_position="position",
        get_text="text",
        get_size=14,
        get_color=[70, 70, 70, 200],
        get_text_anchor='"middle"',
        get_alignment_baseline='"center"',
        pickable=False,
    )

    return pdk.Deck(
        layers=[county_layer, border_layer, label_layer],
        initial_view_state=pdk.ViewState(
            latitude=38.5,
            longitude=-96.5,
            zoom=3.4,
            min_zoom=3,
            max_zoom=9,
        ),
        map_style="light",
        tooltip={"html": "<b>{name}</b><br/>{value}"},
    )


selected_date = st.sidebar.select_slider(
    "Date",
    options=date_options,
    value=date_options[0],
    format_func=lambda d: d.strftime("%b %d"),
)


filtered = df[df["date"].dt.date == selected_date]

st.subheader(f"{label} on {selected_date}")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Counties", f"{len(filtered):,}")
col2.metric("Mean", f"{filtered[selected_variable].mean():.2f} {unit}")
col3.metric("Minimum", f"{filtered[selected_variable].min():.2f} {unit}")
col4.metric("Maximum", f"{filtered[selected_variable].max():.2f} {unit}")

map_col, legend_col = st.columns([12, 1])

with map_col:
    st.pydeck_chart(make_deck(selected_date), height=560)

with legend_col:
    st.markdown(
        legend_html(lut, vmin, vmax, label, unit),
        unsafe_allow_html=True,
    )

st.caption(
    "Daily county-level values; the date slider covers every day of the selected "
    "month. The color scale is fixed per variable across all dates. Gray counties "
    "have no data. Rendering is GPU-accelerated via deck.gl; pan and zoom are "
    "free-form."
)