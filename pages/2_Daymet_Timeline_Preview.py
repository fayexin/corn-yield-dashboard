from pathlib import Path

import streamlit as st


st.set_page_config(
    page_title="Daymet Timeline Preview",
    layout="wide"
)


FRAME_ROOT = Path("data/daymet_frames")


st.title("Daymet Timeline Preview")

st.write(
    "This section displays pre-rendered daily Daymet map frames. "
    "The map is shown as an image, so moving through time is faster than redrawing county polygons."
)


def list_available_frame_sets():
    records = []

    if not FRAME_ROOT.exists():
        return records

    for variable_dir in sorted(FRAME_ROOT.iterdir()):
        if not variable_dir.is_dir():
            continue

        for year_dir in sorted(variable_dir.iterdir()):
            if not year_dir.is_dir():
                continue

            frame_files = sorted(year_dir.glob("*.webp"))

            if frame_files:
                records.append(
                    {
                        "variable": variable_dir.name,
                        "year": year_dir.name,
                        "path": year_dir,
                        "n_frames": len(frame_files),
                    }
                )

    return records


frame_sets = list_available_frame_sets()

if not frame_sets:
    st.warning(
        "No pre-rendered frames were found yet. "
        "Generate frames first with render_daymet_frames.py."
    )
    st.stop()


available_variables = sorted({record["variable"] for record in frame_sets})

selected_variable = st.sidebar.selectbox(
    "Variable",
    available_variables,
)

available_years = sorted(
    {
        record["year"]
        for record in frame_sets
        if record["variable"] == selected_variable
    }
)

selected_year = st.sidebar.selectbox(
    "Year",
    available_years,
    index=len(available_years) - 1,
)

selected_record = [
    record
    for record in frame_sets
    if record["variable"] == selected_variable
    and record["year"] == selected_year
][0]

frame_files = sorted(selected_record["path"].glob("*.webp"))
date_labels = [file.stem for file in frame_files]

selected_date = st.sidebar.select_slider(
    "Date",
    options=date_labels,
    value=date_labels[0],
)

selected_file = selected_record["path"] / f"{selected_date}.webp"

st.subheader(f"{selected_variable} on {selected_date}")

st.image(
    str(selected_file),
    use_container_width=True,
)

st.caption(
    "This timeline uses pre-rendered images. It is designed for smooth visual exploration, "
    "not county-level hover or value lookup. Use the interactive Daymet map page for county-level values."
)