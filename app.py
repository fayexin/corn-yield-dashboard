import streamlit as st


st.set_page_config(
    page_title="Climate and Crop Yield Dashboard",
    layout="wide"
)


st.title("Climate and Crop Yield Visualization Dashboard")

st.write(
    "This dashboard is designed to visualize county-level climate data, hydrologic data, "
    "crop yield change, and deep learning model results. The current version focuses on "
    "county-level daily Daymet visualization across the continental United States."
)

st.divider()

st.header("Available sections")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Daily Daymet Interactive Map")
    st.write(
        "Explore county-level daily Daymet variables by selecting year, month, date, "
        "and variable. This page uses an interactive Plotly map and supports county-level hover values."
    )
    st.caption("Status: available")

with col2:
    st.subheader("Daymet Timeline Preview")
    st.write(
        "View pre-rendered daily Daymet maps through a smooth time-based viewer. "
        "This section is planned for fast visual exploration without county-level hover values."
    )
    st.caption("Status: planned")

col3, col4 = st.columns(2)

with col3:
    st.subheader("Yield Change Visualization")
    st.write(
        "Visualize county-level yield change, model-predicted yield loss, and extreme-weather "
        "sensitivity results."
    )
    st.caption("Status: planned")

with col4:
    st.subheader("GLDAS and Hydrologic Maps")
    st.write(
        "Visualize county-level GLDAS variables, including soil moisture, evapotranspiration, "
        "root-zone moisture, and groundwater storage."
    )
    st.caption("Status: planned")

st.divider()

st.header("Current data coverage")

st.write(
    "The current Daymet section uses county-level daily data stored as monthly Parquet files. "
    "This structure keeps the dashboard lighter because only one month of data is loaded at a time."
)

st.write(
    "Use the sidebar to open the available Daymet interactive map page."
)