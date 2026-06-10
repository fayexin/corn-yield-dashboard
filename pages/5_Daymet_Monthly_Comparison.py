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
    page_title="Daymet Monthly Comparison",
    layout="wide"
)


DATA_PATH = Path("data/daymet_monthly_means.parquet")

BACKGROUND_RGBA = [236, 236, 236, 255]

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
    "tmin": "Blues_r",
    "tmean": "RdYlBu_r",
    "gdd10": "YlOrRd",
    "prcp": "Blues",
    "srad": "YlOrBr",
    "vp": "PuBuGn",
    "vpd": "Plasma",
    "swe": "Blues",
    "dayl": "Viridis",
}


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


@st.cache_data
def load_monthly_means(file_mtime):
    df = pd.read_parquet(DATA_PATH)
    df["fips"] = df["fips"].astype(str).str.zfill(5)
    return df


@st.cache_data
def get_color_range(variable, file_mtime):
    df = load_monthly_means(file_mtime)
    values = df[variable].dropna()
    return (
        float(np.nanpercentile(values, 2)),
        float(np.nanpercentile(values, 98)),
    )


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
    bar_height = 440
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
    <div style="font-size:11px; color:#444; margin-top:4px; max-width:115px;
                overflow-wrap:normal;">
      {label} ({unit})
    </div>
    """


st.title("Daymet Monthly Comparison")

st.write(
    "Compare the same calendar month across two different years, side by side. "
    "Both maps share a fixed color scale, so any difference you see is real — "
    "drought years, wet springs, and decades of change are directly visible. "
    "Hover over a county to read its monthly-mean value."
)


if not DATA_PATH.exists():
    st.error(
        "No monthly-mean data was found. Run build_monthly_means.py to create "
        "data/daymet_monthly_means.parquet."
    )
    st.stop()

means_mtime = DATA_PATH.stat().st_mtime
data = load_monthly_means(means_mtime)

geometries = load_county_geometries()
state_borders, state_labels = load_state_layers()


st.sidebar.header("Controls")

selected_variable = st.sidebar.selectbox(
    "Daymet variable",
    list(VARIABLE_LABELS),
    format_func=lambda variable: f"{VARIABLE_LABELS[variable]} ({variable})",
)

selected_month = st.sidebar.select_slider(
    "Month (shared by both maps)",
    options=list(range(1, 13)),
    value=7,
    format_func=lambda month: month_name[month][:3],
)

available_years = sorted(data["year"].unique())

year_left = st.sidebar.select_slider(
    "Year (left map)",
    options=available_years,
    value=available_years[0],
)

year_right = st.sidebar.select_slider(
    "Year (right map)",
    options=available_years,
    value=available_years[-1],
)


color_low, color_high = get_color_range(selected_variable, means_mtime)
unit = VARIABLE_UNITS[selected_variable]
label = VARIABLE_LABELS[selected_variable]
lut = get_color_lut(VARIABLE_COLOR_SCALES[selected_variable])


def make_deck(year):
    filtered = data[
        (data["year"] == year) & (data["month"] == selected_month)
    ]

    values = dict(zip(filtered["fips"], filtered[selected_variable]))
    names = dict(
        zip(filtered["fips"], filtered["namelsad"] + ", " + filtered["state"])
    )

    for old_fips, new_fips in FIPS_ALIASES.items():
        if new_fips in values:
            values[old_fips] = values[new_fips]
            names[old_fips] = names[new_fips]

    features = build_features(geometries, values, names, lut,
                              color_low, color_high, unit)

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
        get_size=12,
        get_color=[70, 70, 70, 200],
        get_text_anchor='"middle"',
        get_alignment_baseline='"center"',
        pickable=False,
    )

    mean_value = filtered[selected_variable].mean()

    deck = pdk.Deck(
        layers=[county_layer, border_layer, label_layer],
        initial_view_state=pdk.ViewState(
            latitude=38.5,
            longitude=-96.5,
            zoom=3.0,
            min_zoom=3,
            max_zoom=9,
        ),
        map_style="light",
        tooltip={"html": "<b>{name}</b><br/>{value}"},
    )

    return deck, mean_value


st.subheader(f"{label} — {month_name[selected_month]}")

left_col, right_col, legend_col = st.columns([6, 6, 1])

with left_col:
    deck_left, mean_left = make_deck(year_left)
    st.markdown(
        f"**{month_name[selected_month]} {year_left}** — "
        f"CONUS mean {mean_left:.2f} {unit}"
    )
    st.pydeck_chart(deck_left, height=520)

with right_col:
    deck_right, mean_right = make_deck(year_right)
    difference = mean_right - mean_left
    st.markdown(
        f"**{month_name[selected_month]} {year_right}** — "
        f"CONUS mean {mean_right:.2f} {unit} ({difference:+.2f} vs left)"
    )
    st.pydeck_chart(deck_right, height=520)

with legend_col:
    st.markdown(
        legend_html(lut, color_low, color_high, label, unit),
        unsafe_allow_html=True,
    )

st.caption(
    "Values are monthly means per county. Both maps use the same fixed color scale "
    "for the selected variable, so colors are directly comparable between years. "
    "Gray counties have no data. Rendering is GPU-accelerated via deck.gl."
)
