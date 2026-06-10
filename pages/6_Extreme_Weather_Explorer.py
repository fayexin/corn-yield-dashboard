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
    page_title="Extreme Weather Explorer",
    layout="wide"
)


MEANS_PATH = Path("data/daymet_monthly_means.parquet")
NORMALS_PATH = Path("data/daymet_normals.parquet")

BACKGROUND_RGBA = [60, 60, 60, 120]

FIPS_ALIASES = {"46113": "46102", "51515": "51019"}


ANOMALY_CONFIG = {
    "temperature": {
        "column": "tmean",
        "normal": "tmean_normal",
        "label": "Temperature anomaly",
        "unit": "°C",
        "scale_name": "RdBu_r",       # blue = colder, red = warmer
        "elevation_per_unit": 45000,  # meters of column height per °C
    },
    "precipitation": {
        "column": "prcp",
        "normal": "prcp_normal",
        "label": "Precipitation anomaly",
        "unit": "mm/day",
        "scale_name": "BrBG",         # brown = drier, teal = wetter
        "elevation_per_unit": 90000,
    },
}


EVENTS = [
    {
        "name": "July 2012 — Corn Belt drought and heat",
        "year": 2012, "month": 7, "variable": "temperature",
        "blurb": "The most severe US drought since the 1950s; extreme heat "
                 "and dryness devastated Corn Belt yields.",
    },
    {
        "name": "May 2019 — Midwest floods (record wet spring)",
        "year": 2019, "month": 5, "variable": "precipitation",
        "blurb": "Record spring rainfall flooded fields and delayed planting "
                 "across the Midwest.",
    },
    {
        "name": "June 1988 — Drought of 1988",
        "year": 1988, "month": 6, "variable": "temperature",
        "blurb": "One of the costliest US droughts on record, with intense "
                 "early-summer heat across the Plains and Midwest.",
    },
    {
        "name": "July 1993 — Great Mississippi Flood",
        "year": 1993, "month": 7, "variable": "precipitation",
        "blurb": "Persistent rains pushed the Mississippi and Missouri rivers "
                 "into months-long record flooding.",
    },
    {
        "name": "February 2021 — Texas cold wave",
        "year": 2021, "month": 2, "variable": "temperature",
        "blurb": "An Arctic outbreak brought record cold deep into Texas and "
                 "the southern Plains.",
    },
    {
        "name": "August 2017 — Hurricane Harvey",
        "year": 2017, "month": 8, "variable": "precipitation",
        "blurb": "Harvey stalled over southeast Texas, producing the heaviest "
                 "tropical rainfall in US history.",
    },
    {
        "name": "July 2011 — Texas heat and drought",
        "year": 2011, "month": 7, "variable": "temperature",
        "blurb": "The peak of the 2011 Texas drought, the state's hottest "
                 "summer on record.",
    },
    {
        "name": "March 2012 — 'Summer in March' heat wave",
        "year": 2012, "month": 3, "variable": "temperature",
        "blurb": "An extraordinary early-spring heat wave shattered thousands "
                 "of temperature records east of the Rockies.",
    },
]


@st.cache_data
def load_monthly_means():
    df = pd.read_parquet(
        MEANS_PATH,
        columns=["fips", "state", "namelsad", "year", "month", "tmean", "prcp"],
    )
    df["fips"] = df["fips"].astype(str).str.zfill(5)
    return df


@st.cache_data
def load_normals():
    normals = pd.read_parquet(NORMALS_PATH)
    normals["fips"] = normals["fips"].astype(str).str.zfill(5)
    return normals


@st.cache_data
def get_anomaly_range(variable_key):
    config = ANOMALY_CONFIG[variable_key]

    df = load_monthly_means()
    normals = load_normals()

    merged = df.merge(normals, on=["fips", "month"])
    anomaly = merged[config["column"]] - merged[config["normal"]]

    limit = float(np.nanpercentile(anomaly.abs(), 99))

    step = 0.5 if limit < 5 else 1.0
    return max(step, round(limit / step) * step)


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
    bar_height = 420
    span = max(vmax - vmin, 1e-9)

    stops = ", ".join(
        f"rgb({c[0]},{c[1]},{c[2]})" for c in lut[:: max(len(lut) // 24, 1)]
    )

    ticks, decimals = nice_ticks(vmin, vmax)

    tick_items = []
    for tick in ticks:
        bottom = (tick - vmin) / span * bar_height
        text = f"{tick:+.{decimals}f}" if tick else "0"
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
    <div style="font-size:12px; color:#444; margin-top:4px; max-width:100px;">
      {label} ({unit})<br><br>
      Column height = size of the anomaly
    </div>
    """


st.title("Extreme Weather Explorer")

st.write(
    "Every month is compared against each county's 1980–2010 normal for that "
    "calendar month. Columns rise with the size of the departure from normal — "
    "extreme events stand out as mountain ranges. Pick a famous event, or browse "
    "any month since 1980. Drag to rotate, scroll to zoom, hover for values."
)


if not NORMALS_PATH.exists():
    st.error(
        "Climate normals not found. Run build_climate_normals.py to create "
        "data/daymet_normals.parquet."
    )
    st.stop()

data = load_monthly_means()
normals = load_normals()
geometries = load_county_geometries()


st.sidebar.header("Controls")

event_names = [event["name"] for event in EVENTS] + ["Custom month"]

selected_event_name = st.sidebar.selectbox("Event", event_names)

if selected_event_name == "Custom month":
    variable_key = st.sidebar.radio(
        "Variable",
        list(ANOMALY_CONFIG),
        format_func=lambda key: key.capitalize(),
    )
    available_years = sorted(data["year"].unique())
    selected_year = st.sidebar.select_slider(
        "Year", options=available_years, value=available_years[-1]
    )
    selected_month = st.sidebar.select_slider(
        "Month",
        options=list(range(1, 13)),
        value=7,
        format_func=lambda month: month_name[month][:3],
    )
    blurb = None
else:
    event = next(e for e in EVENTS if e["name"] == selected_event_name)
    variable_key = event["variable"]
    selected_year = event["year"]
    selected_month = event["month"]
    blurb = event["blurb"]

exaggeration = st.sidebar.slider(
    "Height exaggeration", 0.5, 3.0, 1.0, step=0.5
)

config = ANOMALY_CONFIG[variable_key]

selected = data[
    (data["year"] == selected_year) & (data["month"] == selected_month)
]

if selected.empty:
    st.info(
        f"No data for {month_name[selected_month]} {selected_year}. If the "
        "monthly means file does not yet cover the full record, rebuild it "
        "from the 1980–2025 data."
    )
    st.stop()

merged = selected.merge(
    normals[["fips", "month", config["normal"]]], on=["fips", "month"]
)
merged["anomaly"] = merged[config["column"]] - merged[config["normal"]]

limit = get_anomaly_range(variable_key)
lut = get_color_lut(config["scale_name"])
unit = config["unit"]


st.subheader(
    f"{config['label']} — {month_name[selected_month]} {selected_year}"
)

if blurb:
    st.markdown(f"*{blurb}*")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Counties", f"{len(merged):,}")
col2.metric("Mean anomaly", f"{merged['anomaly'].mean():+.2f} {unit}")
col3.metric("Most negative", f"{merged['anomaly'].min():+.2f} {unit}")
col4.metric("Most positive", f"{merged['anomaly'].max():+.2f} {unit}")


anomalies = dict(zip(merged["fips"], merged["anomaly"]))
actuals = dict(zip(merged["fips"], merged[config["column"]]))
norms = dict(zip(merged["fips"], merged[config["normal"]]))
names = dict(zip(merged["fips"], merged["namelsad"] + ", " + merged["state"]))

for old_fips, new_fips in FIPS_ALIASES.items():
    if new_fips in anomalies:
        anomalies[old_fips] = anomalies[new_fips]
        actuals[old_fips] = actuals[new_fips]
        norms[old_fips] = norms[new_fips]
        names[old_fips] = names[new_fips]


span = 2 * limit
features = []

for county in geometries:
    fips = county["fips"]
    anomaly = anomalies.get(fips)

    if anomaly is None or pd.isna(anomaly):
        fill = BACKGROUND_RGBA
        elevation = 0
        tooltip_value = "No data"
    else:
        index = int(np.clip((anomaly + limit) / span, 0.0, 1.0) * 255)
        fill = lut[index] + [235]
        elevation = (
            abs(float(anomaly))
            * config["elevation_per_unit"]
            * exaggeration
        )
        tooltip_value = (
            f"Anomaly: {anomaly:+.2f} {unit}<br/>"
            f"Observed: {actuals[fips]:.2f} {unit.replace('°C', '°C')}<br/>"
            f"Normal: {norms[fips]:.2f}"
        )

    features.append(
        {
            "type": "Feature",
            "geometry": county["geometry"],
            "properties": {
                "name": names.get(fips, f"County {fips}"),
                "value": tooltip_value,
                "fill": fill,
                "elevation": elevation,
            },
        }
    )


layer = pdk.Layer(
    "GeoJsonLayer",
    data={"type": "FeatureCollection", "features": features},
    extruded=True,
    get_elevation="properties.elevation",
    get_fill_color="properties.fill",
    get_line_color=[30, 30, 30, 80],
    line_width_min_pixels=0.3,
    pickable=True,
    stroked=False,
    filled=True,
    wireframe=False,
)

deck = pdk.Deck(
    layers=[layer],
    initial_view_state=pdk.ViewState(
        latitude=37.5,
        longitude=-96.5,
        zoom=3.3,
        pitch=50,
        bearing=0,
        min_zoom=3,
        max_zoom=9,
    ),
    map_style="dark",
    tooltip={"html": "<b>{name}</b><br/>{value}"},
)

map_col, legend_col = st.columns([12, 1])

with map_col:
    st.pydeck_chart(deck, height=620)

with legend_col:
    st.markdown(
        legend_html(lut, -limit, limit, config["label"], unit),
        unsafe_allow_html=True,
    )

st.caption(
    "Anomalies are departures from each county's 1980–2010 average for the same "
    "calendar month. Column height shows the size of the anomaly; color shows its "
    "direction. Hold Ctrl (or right-click) and drag to tilt and rotate the view."
)
