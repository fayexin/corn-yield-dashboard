"""
Render an MP4 animation of county-level yield across years, one per crop.

Frames are drawn offline with geopandas/matplotlib and encoded to MP4, so the
Streamlit app only plays a small video file (no live animation, no extra
runtime dependencies).

Setup (one-time, on your machine -- not on Streamlit Cloud):
    conda install -c conda-forge geopandas matplotlib imageio imageio-ffmpeg
    python render_yield_video.py

Output:
    data/yield_video_corn.mp4
    data/yield_video_soybeans.mp4
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import geopandas as gpd
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from shapely.geometry import shape


DATA_PATH = Path("data/crop_yield_county.csv")
GEOJSON_PATH = Path("data/geo/geojson-counties-fips.json")

CROP_LABELS = {"corn": "Corn", "soybeans": "Soybeans"}

YIELD_COLOR_SCALE = "YlGn"
BACKGROUND_FILL = "#ececec"
FPS = 4

# Counties outside the continental US (Alaska, Hawaii, territories).
NON_CONUS_STATE_FIPS = {"02", "15", "60", "66", "69", "72", "78"}

# EPSG code for CONUS Albers Equal Area, for a clean undistorted map.
CONUS_ALBERS = 5070


def load_counties():
    with open(GEOJSON_PATH) as handle:
        geojson = json.load(handle)

    records = [
        {"fips": feature["id"], "geometry": shape(feature["geometry"])}
        for feature in geojson["features"]
    ]

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    gdf = gdf[~gdf["fips"].str[:2].isin(NON_CONUS_STATE_FIPS)]
    gdf = gdf.to_crs(epsg=CONUS_ALBERS)
    gdf["state"] = gdf["fips"].str[:2]

    return gdf


def render_crop(crop, yields, counties, state_lines, bounds):
    values = yields.loc[yields["crop"] == crop]

    vmin = float(np.nanpercentile(values["yield_bu_acre"], 2))
    vmax = float(np.nanpercentile(values["yield_bu_acre"], 98))
    norm = Normalize(vmin=vmin, vmax=vmax)

    years = sorted(values["year"].unique())
    out_path = Path(f"data/yield_video_{crop}.mp4")

    writer = imageio.get_writer(out_path, fps=FPS, macro_block_size=16)

    for year in years:
        year_values = values.loc[
            values["year"] == year, ["fips", "yield_bu_acre"]
        ]
        frame_gdf = counties.merge(year_values, on="fips", how="left")

        fig, ax = plt.subplots(figsize=(9, 6), dpi=120)

        frame_gdf.plot(
            column="yield_bu_acre",
            cmap=YIELD_COLOR_SCALE,
            norm=norm,
            ax=ax,
            linewidth=0.05,
            edgecolor="0.6",
            missing_kwds={
                "color": BACKGROUND_FILL,
                "edgecolor": "0.6",
                "linewidth": 0.05,
            },
        )

        state_lines.plot(ax=ax, color="black", linewidth=0.5)

        ax.set_xlim(bounds[0], bounds[2])
        ax.set_ylim(bounds[1], bounds[3])
        ax.set_axis_off()
        ax.set_title(f"{CROP_LABELS[crop]} yield in {year}", fontsize=15)

        scalar_mappable = ScalarMappable(norm=norm, cmap=YIELD_COLOR_SCALE)
        colorbar = fig.colorbar(scalar_mappable, ax=ax, shrink=0.6, pad=0.01)
        colorbar.set_label("Yield (bu/acre)")

        fig.tight_layout()
        fig.canvas.draw()

        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        writer.append_data(frame)

        plt.close(fig)
        print(f"{crop} {year}")

    writer.close()
    print(f"Saved {out_path}")


def main():
    yields = pd.read_csv(DATA_PATH, dtype={"fips": str})
    yields["fips"] = yields["fips"].str.zfill(5)

    counties = load_counties()
    state_lines = counties.dissolve(by="state").boundary
    bounds = counties.total_bounds

    for crop in CROP_LABELS:
        if crop in yields["crop"].unique():
            render_crop(crop, yields, counties, state_lines, bounds)


if __name__ == "__main__":
    main()