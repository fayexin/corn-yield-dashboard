import json
from calendar import month_name
from pathlib import Path
from urllib.request import urlopen

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="County-Level Daymet Dashboard",
    layout="wide"
)


DATA_DIR = Path("data/daymet_monthly")


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


st.title("County-Level Daymet Daily Visualization")

st.write(
    "This dashboard visualizes county-level daily Daymet variables across the continental United States. "
    "Select a year, month, date, and variable to view daily spatial patterns."
)


file_index = list_available_files()

if file_index.empty:
    st.error(
        "No Parquet files were found. Please check that files exist in data/daymet_monthly."
    )
    st.stop()


counties = load_county_geojson()
states = load_state_geojson()


st.sidebar.header("Controls")

available_years = sorted(file_index["year"].unique())

selected_year = st.sidebar.selectbox(
    "Year",
    available_years,
    index=len(available_years) - 1
)

available_months = sorted(
    file_index.loc[file_index["year"] == selected_year, "month"].unique()
)

selected_month = st.sidebar.selectbox(
    "Month",
    available_months,
    format_func=lambda month: f"{month:02d} - {month_name[month]}"
)

selected_file = file_index.loc[
    (file_index["year"] == selected_year)
    & (file_index["month"] == selected_month),
    "file"
].iloc[0]

df = load_month_data(selected_file)

available_variables = [
    variable for variable in VARIABLE_LABELS
    if variable in df.columns
]

selected_variable = st.sidebar.selectbox(
    "Daymet variable",
    available_variables,
    format_func=lambda variable: f"{VARIABLE_LABELS[variable]} ({variable})"
)

state_label_mode = st.sidebar.selectbox(
    "State labels",
    ["None", "Abbreviation", "Full name"]
)

date_options = sorted(df["date"].dt.date.unique())

if len(date_options) == 1:
    selected_date = date_options[0]
    st.sidebar.write(f"Date: {selected_date}")
else:
    selected_date = st.sidebar.select_slider(
        "Date",
        options=date_options,
        value=date_options[0]
    )


filtered = df[df["date"].dt.date == selected_date].copy()

unit = VARIABLE_UNITS[selected_variable]


st.subheader(
    f"{VARIABLE_LABELS[selected_variable]} on {selected_date}"
)

col1, col2, col3, col4 = st.columns(4)

col1.metric("Counties", f"{len(filtered):,}")
col2.metric("Mean", f"{filtered[selected_variable].mean():.2f} {unit}")
col3.metric("Minimum", f"{filtered[selected_variable].min():.2f} {unit}")
col4.metric("Maximum", f"{filtered[selected_variable].max():.2f} {unit}")


fig = px.choropleth(
    filtered,
    geojson=counties,
    locations="fips",
    color=selected_variable,
    scope="usa",
    hover_name="namelsad",
    hover_data={
        "state": True,
        "fips": True,
        selected_variable: ":.2f",
    },
    custom_data=[
        "fips",
        "state",
        "county",
        "namelsad",
        selected_variable,
    ],
    color_continuous_scale=VARIABLE_COLOR_SCALES[selected_variable],
    labels={
        selected_variable: f"{VARIABLE_LABELS[selected_variable]} ({unit})"
    },
)


available_states = set(filtered["state"].unique())

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
    projection_scale=1.18
)

fig.update_coloraxes(
    colorbar=dict(
        title=dict(
            text=f"{VARIABLE_LABELS[selected_variable]}<br>({unit})",
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
    key="daymet_county_map"
)

st.caption(
    "Tip: hover over a county to view county name, FIPS, state, and the selected Daymet value."
)