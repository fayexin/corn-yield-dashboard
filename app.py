from pathlib import Path

import streamlit as st


st.set_page_config(
    page_title="Climate and Crop Yield Dashboard",
    layout="wide"
)


CROP_LABELS = {"corn": "Corn", "soybeans": "Soybeans"}


@st.cache_data
def load_video_bytes(path):
    return Path(path).read_bytes()


st.title("Climate and Crop Yield Visualization Dashboard")

st.write(
    "This dashboard is designed to visualize county-level climate data, hydrologic data, "
    "crop yield, and deep learning model results across the continental United States."
)

st.divider()

st.header("Yield over time")

st.write(
    "County-level corn and soybean yields, animated for every year on record. "
    "Select a crop to switch the animation."
)

selected_crop = st.segmented_control(
    "Crop",
    options=list(CROP_LABELS.keys()),
    format_func=lambda crop: CROP_LABELS[crop],
    default="corn",
)

if selected_crop is None:
    selected_crop = "corn"

video_path = Path(f"data/yield_video_{selected_crop}.mp4")

if video_path.exists():
    st.video(
        load_video_bytes(str(video_path)),
        loop=True,
        autoplay=True,
        muted=True,
    )
else:
    st.info(
        "The yield animation has not been generated yet. Run render_yield_video.py "
        "to create data/yield_video_corn.mp4 and data/yield_video_soybeans.mp4."
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
    st.subheader("County-Level Crop Yield")
    st.write(
        "Visualize county-level corn and soybean yields by year, with a fixed color scale "
        "for comparison across years and an all-county background for context."
    )
    st.caption("Status: available")

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
    "The Daymet sections use county-level daily data stored as monthly Parquet files, "
    "which keeps the dashboard lighter because only one month is loaded at a time. "
    "The yield section uses USDA NASS county-level corn and soybean estimates."
)

st.write(
    "Use the sidebar to open any available page."
)