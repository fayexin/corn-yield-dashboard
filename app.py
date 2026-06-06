import streamlit as st

st.set_page_config(
    page_title="Corn Yield Dashboard",
    layout="wide"
)

st.title("County-Level Corn Yield Prediction Dashboard")

st.write(
    "This dashboard will show county-level climate variables, deep learning model results, "
    "and extreme weather sensitivity analysis."
)

st.subheader("Planned sections")

st.write("1. County-level Daymet map")
st.write("2. Model prediction results")
st.write("3. Predicted versus observed yield")
st.write("4. Extreme weather scenario explorer")

st.info("This is the first test version of the dashboard.")