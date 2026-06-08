import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Climate and Crop Yield Dashboard",
    layout="wide"
)


DATA_PATH = Path("data/crop_yield_county.csv")

CROP_LABELS = {"corn": "Corn", "soybeans": "Soybeans"}

YIELD_COLOR_SCALE = "YlGn"
BACKGROUND_FILL = "#ececec"


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
def load_yield_data():
    df = pd.read_csv(DATA_PATH, dtype={"fips": str})
    df["fips"] = df["fips"].str.zfill(5)
    return df


@st.cache_data
def get_color_range(crop):
    df = load_yield_data()
    values = df.loc[df["crop"] == crop, "yield_bu_acre"].dropna()
    return (
        float(np.nanpercentile(values, 2)),
        float(np.nanpercentile(values, 98)),
    )


@st.cache_data
def load_county_geojson():
    url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
    with urlopen(url) as response:
        return json.load(response)


@st.cache_data
def get_all_county_fips(counties_geojson):
    return [feature["id"] for feature in counties_geojson["features"]]


@st.cache_data
def load_state_geojson():
    url = "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
    with urlopen(url) as response:
        return json.load(response)


def extract_state_boundary_lines(states_geojson, available_states):
    border_lons = []
    border_lats = []

    for feature in states_geojson["features"]:
        state_abbr = STATE_NAME_TO_ABBR.get(feature["properties"]["name"])

        if state_abbr not in available_states:
            continue

        geometry = feature["geometry"]

        if geometry["type"] == "Polygon":
            polygons = [geometry["coordinates"]]
        elif geometry["type"] == "MultiPolygon":
            polygons = geometry["coordinates"]
        else:
            continue

        for polygon in polygons:
            outer_ring = polygon[0]
            border_lons.extend([point[0] for point in outer_ring])
            border_lats.extend([point[1] for point in outer_ring])
            border_lons.append(None)
            border_lats.append(None)

    return border_lons, border_lats


st.title("Climate and Crop Yield Visualization Dashboard")

st.write(
    "This dashboard visualizes county-level climate data, hydrologic data, crop yield, "
    "and deep learning model results across the continental United States."
)

st.divider()

st.header("County-level crop yield")

data = load_yield_data()

if data.empty:
    st.error(
        "No yield data was found. Please check that data/crop_yield_county.csv exists."
    )
    st.stop()

counties = load_county_geojson()
states = load_state_geojson()
all_county_fips = get_all_county_fips(counties)

control_left, control_right = st.columns([1, 2])

with control_left:
    selected_crop = st.segmented_control(
        "Crop",
        options=list(CROP_LABELS),
        format_func=lambda crop: CROP_LABELS[crop],
        default="corn",
    )

if selected_crop is None:
    selected_crop = "corn"

crop_data = data[data["crop"] == selected_crop]
available_years = sorted(crop_data["year"].unique())

with control_right:
    selected_year = st.select_slider(
        "Year",
        options=available_years,
        value=available_years[-1],
    )

filtered = crop_data[crop_data["year"] == selected_year].copy()
color_low, color_high = get_color_range(selected_crop)
crop_label = CROP_LABELS[selected_crop]

st.subheader(f"{crop_label} yield in {selected_year}")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Counties", f"{len(filtered):,}")
col2.metric("Median", f"{filtered['yield_bu_acre'].median():.1f} bu/acre")
col3.metric("Minimum", f"{filtered['yield_bu_acre'].min():.1f} bu/acre")
col4.metric("Maximum", f"{filtered['yield_bu_acre'].max():.1f} bu/acre")

fig = go.Figure()

fig.add_trace(
    go.Choropleth(
        geojson=counties,
        locations=all_county_fips,
        z=[0] * len(all_county_fips),
        colorscale=[[0, BACKGROUND_FILL], [1, BACKGROUND_FILL]],
        showscale=False,
        marker_line_color="rgba(120, 120, 120, 0.35)",
        marker_line_width=0.15,
        hoverinfo="skip",
    )
)

fig.add_trace(
    go.Choropleth(
        geojson=counties,
        locations=filtered["fips"],
        z=filtered["yield_bu_acre"],
        zmin=color_low,
        zmax=color_high,
        colorscale=YIELD_COLOR_SCALE,
        marker_line_color="rgba(90, 90, 90, 0.30)",
        marker_line_width=0.10,
        customdata=filtered[["county_name", "state_alpha", "fips"]].to_numpy(),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "State: %{customdata[1]}<br>"
            "FIPS: %{customdata[2]}<br>"
            "Yield: %{z:.1f} bu/acre<extra></extra>"
        ),
        colorbar=dict(
            title=dict(text="Yield<br>(bu/acre)", side="right"),
            x=1.02, xanchor="left", y=0.50, yanchor="middle",
            len=0.80, thickness=18,
        ),
    )
)

available_states = set(filtered["state_alpha"].unique())
state_border_lons, state_border_lats = extract_state_boundary_lines(
    states, available_states
)

fig.add_trace(
    go.Scattergeo(
        lon=state_border_lons,
        lat=state_border_lats,
        mode="lines",
        line=dict(width=1.2, color="rgba(40, 40, 40, 0.75)"),
        hoverinfo="skip",
        showlegend=False,
    )
)

fig.update_geos(
    visible=False,
    projection_type="albers usa",
    lonaxis_range=[-125, -66],
    lataxis_range=[24, 50],
    projection_scale=0.92,
)

fig.update_layout(
    height=820,
    margin={"r": 90, "t": 20, "l": 0, "b": 0},
    geo=dict(domain=dict(x=[0.00, 0.96], y=[0.00, 1.00])),
)

st.plotly_chart(fig, use_container_width=True, key="home_yield_map")

st.divider()

st.header("Available sections")

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Daily Daymet Interactive Map")
    st.write(
        "Explore county-level daily Daymet variables by selecting year, month, date, "
        "and variable, on an interactive map with county-level hover values."
    )
    st.caption("Status: available")

with col_b:
    st.subheader("Daymet Monthly Timeline")
    st.write(
        "Scrub a month slider through the record to view county-level monthly-mean "
        "Daymet variables across the continental United States."
    )
    st.caption("Status: available")

col_c, col_d = st.columns(2)

with col_c:
    st.subheader("County-Level Crop Yield")
    st.write(
        "Visualize county-level corn and soybean yields by year, with a fixed color "
        "scale for comparison across years and an all-county background for context."
    )
    st.caption("Status: available")

with col_d:
    st.subheader("GLDAS and Hydrologic Maps")
    st.write(
        "Visualize county-level GLDAS variables, including soil moisture, "
        "evapotranspiration, root-zone moisture, and groundwater storage."
    )
    st.caption("Status: planned")

st.divider()

st.write(
    "Use the sidebar to open any available page."
)