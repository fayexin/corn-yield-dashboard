import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Corn Yield Dashboard",
    layout="wide"
)

st.title("County-Level Corn Yield Prediction Dashboard")

st.write(
    "This dashboard shows county-level climate variables, deep learning model results, "
    "and extreme weather sensitivity analysis."
)

@st.cache_data
def load_model_summary():
    df = pd.read_csv("data/model_summary.csv")
    return df

model_summary = load_model_summary()

st.subheader("Model comparison")

st.write(
    "This test chart compares model performance using sample R² and RMSE values. "
    "Later, this file can be replaced with the final dissertation results."
)

col1, col2 = st.columns(2)

with col1:
    fig_r2 = px.bar(
        model_summary,
        x="model",
        y="r2",
        title="Model comparison by R²",
        text="r2",
        labels={
            "model": "Model",
            "r2": "R²"
        }
    )

    fig_r2.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_r2.update_layout(yaxis_range=[0, 1])
    st.plotly_chart(fig_r2, use_container_width=True)

with col2:
    fig_rmse = px.bar(
        model_summary,
        x="model",
        y="rmse",
        title="Model comparison by RMSE",
        text="rmse",
        labels={
            "model": "Model",
            "rmse": "RMSE"
        }
    )

    fig_rmse.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    st.plotly_chart(fig_rmse, use_container_width=True)

st.subheader("Data table")

st.dataframe(model_summary, use_container_width=True)

st.info("This is still a test version. The next step is to add county-level data.")