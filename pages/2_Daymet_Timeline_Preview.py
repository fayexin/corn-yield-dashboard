import streamlit as st

st.set_page_config(
    page_title="Daymet Timeline Preview",
    layout="wide"
)

st.title("Daymet Timeline Preview")

st.write(
    "This section will show pre-rendered daily Daymet maps as a smooth time-based viewer. "
    "Unlike the interactive map, this page will not redraw county polygons each time. "
    "It will display pre-generated map frames for faster daily scrolling."
)

st.info("This section is planned. The next step is to generate daily map frames for one variable and one year.")