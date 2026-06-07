import json
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Yield Change Visualization",
    layout="wide"
)


DATA_PATH = Path("data/crop_yield_county_1980_present.csv")


CROP_LABELS = {
    "corn": "Corn",
    "soybeans": "Soybeans",
}


# Diverging scale: loss = red, no change = white, gain = blue.
CHANGE_COLOR_SCALE = [
    [0.0, "#b2182b"],
    [0.5, "#f7f7f7"],
    [1.0, "#2166ac"],
]


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


@st.cache_data
def load_change_data():
    df = pd.read_csv(DATA_PATH, dtype={"fips": str})

    df["fips"] = df["fips"].str.zfill(5)
    df = df.sort_values(["crop", "fips", "year"])

    grouped = df.groupby(["crop", "fips"])
    df["yield_prev"] = grouped["yield_bu_acre"].shift(1)
    df["prev_year"] = grouped["year"].shift(1)

    # Only treat as a year-over-year change when years are consecutive,
    # so disclosure-suppressed gaps do not produce misleading jumps.
    consecutive = df["prev_year"] == (df["year"] - 1)
    df["change"] = df["yield_bu_acre"] - df["yield_prev"]
    df.loc[~consecutive, "change"] = np.nan

    return df


@st.cache_data
def get_color_limit(crop):
    df = load_change_data()
    crop_changes = df.loc[df["crop"] == crop, "change"].dropna()

    # 90th percentile of absolute change keeps the typical signal readable
    # while letting the most extreme counties saturate at the scale ends.
    limit = float(np.nanpercentile(crop_changes.abs(), 90))

    return max(5.0, round(limit / 5.0) * 5.0)


@st.cache_data
def load_county_geojson():
    url = "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"

    with urlopen(url) as response:
        counties = json.load(response)

    return counties


@st.cache_data
def load_state_geojson():
    url = "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"

    with urlopen(url) as response:
        states = json.load(response)

    return states


def extract_state_boundary_lines(states_geojson, available_states):
    border_lons = []
    border_lats = []

    for feature in states_geojson["features"]:
        state_name = feature["properties"]["name"]
        state_abbr = STATE_NAME_TO_ABBR.get(state_name)

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


def get_state_label_points(states_geojson, available_states):
    label_rows = []

    for feature in states_geojson["features"]:
        state_name = feature["properties"]["name"]
        state_abbr = STATE_NAME_TO_ABBR.get(state_name)

        if state_abbr not in available_states:
            continue

        geometry = feature["geometry"]

        if geometry["type"] == "Polygon":
            polygons = [geometry["coordinates"]]
        elif geometry["type"] == "MultiPolygon":
            polygons = geometry["coordinates"]
        else:
            continue

        all_lons = []
        all_lats = []

        for polygon in polygons:
            outer_ring = polygon[0]
            all_lons.extend([point[0] for point in outer_ring])
            all_lats.extend([point[1] for point in outer_ring])

        if all_lons and all_lats:
            label_rows.append(
                {
                    "state": state_abbr,
                    "state_name": state_name,
                    "lon": sum(all_lons) / len(all_lons),
                    "lat": sum(all_lats) / len(all_lats),
                }
            )

    return pd.DataFrame(label_rows)


st.title("County-Level Year-over-Year Yield Change")

st.write(
    "This page shows the year-over-year change in county-level corn and soybean yields "
    "across the continental United States, based on USDA NASS survey estimates. "
    "Select a crop and year to compare each county's yield against the previous year. "
    "Red counties lost yield relative to the prior year; blue counties gained."
)


data = load_change_data()

if data.empty:
    st.error(
        "No yield data was found. Please check that "
        "data/crop_yield_county_1980_present.csv exists."
    )
    st.stop()


counties = load_county_geojson()
states = load_state_geojson()


st.sidebar.header("Controls")

available_crops = [crop for crop in CROP_LABELS if crop in data["crop"].unique()]

selected_crop = st.sidebar.segmented_control(
    "Crop",
    options=available_crops,
    format_func=lambda crop: CROP_LABELS[crop],
    default=available_crops[0],
)

if selected_crop is None:
    selected_crop = available_crops[0]


crop_data = data[data["crop"] == selected_crop]

change_years = sorted(
    crop_data.loc[crop_data["change"].notna(), "year"].unique()
)

selected_year = st.sidebar.select_slider(
    "Change year (vs. previous year)",
    options=change_years,
    value=change_years[-1],
)

state_label_mode = st.sidebar.selectbox(
    "State labels",
    ["None", "Abbreviation", "Full name"]
)


filtered = crop_data[
    (crop_data["year"] == selected_year) & crop_data["change"].notna()
].copy()

color_limit = get_color_limit(selected_crop)
crop_label = CROP_LABELS[selected_crop]


st.subheader(
    f"{crop_label} yield change: {selected_year - 1} to {selected_year}"
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Counties", f"{len(filtered):,}")
col2.metric("Median change", f"{filtered['change'].median():+.1f} bu/acre")
col3.metric("Largest gain", f"{filtered['change'].max():+.1f} bu/acre")
col4.metric("Largest loss", f"{filtered['change'].min():+.1f} bu/acre")


fig = px.choropleth(
    filtered,
    geojson=counties,
    locations="fips",
    color="change",
    scope="usa",
    hover_name="county_name",
    hover_data={
        "state_alpha": True,
        "fips": True,
        "yield_prev": ":.1f",
        "yield_bu_acre": ":.1f",
        "change": ":+.1f",
    },
    color_continuous_scale=CHANGE_COLOR_SCALE,
    range_color=[-color_limit, color_limit],
    labels={
        "change": "Change (bu/acre)",
        "yield_prev": f"{selected_year - 1} yield",
        "yield_bu_acre": f"{selected_year} yield",
        "state_alpha": "State",
    },
)


available_states = set(filtered["state_alpha"].unique())

state_border_lons, state_border_lats = extract_state_boundary_lines(
    states,
    available_states
)

fig.add_trace(
    go.Scattergeo(
        lon=state_border_lons,
        lat=state_border_lats,
        mode="lines",
        line=dict(
            width=1.2,
            color="rgba(40, 40, 40, 0.75)"
        ),
        hoverinfo="skip",
        showlegend=False
    )
)

if state_label_mode != "None":
    state_labels = get_state_label_points(states, available_states)

    if not state_labels.empty:
        if state_label_mode == "Abbreviation":
            label_text = state_labels["state"]
            text_size = 12
        else:
            label_text = state_labels["state_name"]
            text_size = 9

        fig.add_trace(
            go.Scattergeo(
                lon=state_labels["lon"],
                lat=state_labels["lat"],
                mode="text",
                text=label_text,
                textfont=dict(
                    size=text_size,
                    color="rgba(80, 80, 80, 0.45)"
                ),
                hoverinfo="skip",
                showlegend=False
            )
        )


fig.update_traces(
    marker_line_width=0.02,
    marker_line_color="rgba(90, 90, 90, 0.18)",
    selector=dict(type="choropleth")
)

fig.update_geos(
    visible=False,
    projection_type="albers usa",
    lonaxis_range=[-125, -66],
    lataxis_range=[24, 50],
    projection_scale=0.92
)

fig.update_coloraxes(
    colorbar=dict(
        title=dict(
            text="Change<br>(bu/acre)",
            side="right"
        ),
        x=1.02,
        xanchor="left",
        y=0.50,
        yanchor="middle",
        len=0.80,
        thickness=18
    )
)

fig.update_layout(
    height=850,
    margin={"r": 90, "t": 20, "l": 0, "b": 0},
    geo=dict(
        domain=dict(x=[0.00, 0.96], y=[0.00, 1.00])
    )
)


st.plotly_chart(
    fig,
    use_container_width=True,
    key="yield_change_map"
)

st.caption(
    "Color scale is fixed across years for each crop, so a given shade means the same "
    "change everywhere. Hover over a county to see both years' yields and the change. "
    "Counties suppressed by NASS for disclosure reasons appear blank."
)