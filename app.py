import streamlit as st
import numpy as np
import tensorflow as tf
import xarray as xr
import plotly.express as px
from sklearn.linear_model import LinearRegression
import os
import urllib
import tarfile
import pandas as pd
import xarray as xr
from eofs.xarray import Eof
import gdown

# from esem import gp_model
import geopandas as gpd
import streamlit as st
import time
import urllib.request
import tarfile
import os
import plotly.graph_objects as go
import matplotlib.pyplot as plt

max_co2 = 9500


def year_ui():
    st.sidebar.markdown("# Year")
    year = st.sidebar.slider("Choose a year:", 2015, 2100, 2025, 1)
    return year


# Sidebar for emissions input
def emissions_ui():
    st.sidebar.markdown("# Emissions")
    co2 = st.sidebar.slider("Cumulative CO2 Amount (GtCO2)", 3340, max_co2, 6420, 10)
    return co2


emulator_colors = {
    "Pattern Scaling": "#d55e00",
    "CNN-LTSM": "#cc79a7",
    "Random Forest": "#009e73",
    "Gaussian Process": "#0072b2",
}


def emulator_ui():
    st.sidebar.markdown("# Select Emulator")

    selected_emulator = st.sidebar.radio(
        "Choose an emulator:",
        list(emulator_colors.keys()),
        index=3,  # Default selection
    )

    # Show selected emulator in a color-coded box
    color = emulator_colors[selected_emulator]

    return selected_emulator, {selected_emulator: color}


import numpy as np
import plotly.graph_objects as go
import streamlit as st


def line_plot(df, year):
    # Mask for showing past vs. future data
    mask_past = df[df["year"] <= year]
    mask_future = df[df["year"] > year]

    # Create the plot
    fig = go.Figure()

    # --- Plot Median Line for Past ---
    fig.add_trace(
        go.Scatter(
            x=mask_past["year"],
            y=mask_past["50q_dH_dT"],
            mode="lines",
            name="Median Projection",
            line=dict(color="blue", width=2),
        )
    )

    # --- Plot Uncertainty Bands (5th-95th and 17th-83rd) for Past ---
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([mask_past["year"], mask_past["year"][::-1]]),
            y=np.concatenate([mask_past["95q_dH_dT"], mask_past["5q_dH_dT"][::-1]]),
            fill="toself",
            fillcolor="rgba(0, 0, 255, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="90% Uncertainty (5th-95th)",
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=np.concatenate([mask_past["year"], mask_past["year"][::-1]]),
            y=np.concatenate([mask_past["83q_dH_dT"], mask_past["17q_dH_dT"][::-1]]),
            fill="toself",
            fillcolor="rgba(0, 0, 255, 0.3)",
            line=dict(color="rgba(255,255,255,0)"),
            name="66% Uncertainty (17th-83rd)",
            hoverinfo="skip",
        )
    )

    # --- Gray Out Future Data ---
    fig.add_trace(
        go.Scatter(
            x=mask_future["year"],
            y=mask_future["50q_dH_dT"],
            mode="lines",
            name="Future Projection",
            line=dict(color="gray", dash="dot", width=2),
            hoverinfo="skip",
        )
    )

    # --- Gray Out Future Uncertainty ---
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([mask_future["year"], mask_future["year"][::-1]]),
            y=np.concatenate([mask_future["95q_dH_dT"], mask_future["5q_dH_dT"][::-1]]),
            fill="toself",
            fillcolor="rgba(169, 169, 169, 0.2)",  # Light gray fill
            line=dict(color="rgba(255,255,255,0)"),
            name="Future 90% Uncertainty",
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=np.concatenate([mask_future["year"], mask_future["year"][::-1]]),
            y=np.concatenate(
                [mask_future["83q_dH_dT"], mask_future["17q_dH_dT"][::-1]]
            ),
            fill="toself",
            fillcolor="rgba(169, 169, 169, 0.3)",  # Light gray fill
            line=dict(color="rgba(255,255,255,0)"),
            name="Future 66% Uncertainty",
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        # title=dict(
        #     text="Sea Level Rise Projection with Uncertainty",  # Title Text
        #     font=dict(color="black")  # Title text color
        # ),
        xaxis=dict(
            title="Year",  # X-axis title
            title_font=dict(color="black"),  # X-axis title color
            tickfont=dict(color="black"),  # X-axis tick labels color
            range=[2015, 2100],
        ),
        yaxis=dict(
            title="Sea Level (mm)",  # Y-axis title
            title_font=dict(color="black"),  # Y-axis title color
            tickfont=dict(color="black"),  # Y-axis tick labels color
            range=[df["5q_dH_dT"].min() - 5, df["95q_dH_dT"].max() + 5],
        ),
        showlegend=True,
        legend=dict(
            font=dict(color="black"),  # Legend text color
            # bgcolor="white",            # Legend background color
            # bordercolor="white",        # Border color
            # borderwidth=2               # Border thickness
        ),
        paper_bgcolor="white",  # Outer background (outside the graph)
        plot_bgcolor="white",  # Inner plot background (inside the graph)
    )

    st.plotly_chart(fig)


def plot_horizontal_boxplot(quartiles, emulator):
    """
    Creates a horizontal box plot showing sea level rise quartiles.
    - 5th to 95th percentile is the range.
    - Median (50th percentile) is clearly marked.
    - 17th and 83rd percentiles are shown below the box plot with dotted lines.
    - Uses iPhones for size reference (1 iPhone = 7.8 mm).
    """
    fig, ax = plt.subplots(figsize=(8, 2))  # Wide and short for readability

    percentiles = [
        quartiles["5q_dH_dT"],
        quartiles["17q_dH_dT"],
        quartiles["50q_dH_dT"],
        quartiles["83q_dH_dT"],
        quartiles["95q_dH_dT"],
    ]

    ax.boxplot(
        percentiles,
        vert=False,
        patch_artist=True,
        boxprops=dict(facecolor="lightgrey", color=emulator_colors[emulator]),
        medianprops=dict(color=emulator_colors[emulator], linewidth=2),
        whiskerprops=dict(color=emulator_colors[emulator], linewidth=1),
        capprops=dict(color=emulator_colors[emulator], linewidth=1),
        flierprops=dict(marker="o", color=emulator_colors[emulator], alpha=0.5),
    )

    # Convert mm to iPhone thickness equivalents
    iphones = {name: quartiles[name] / 146.6 for name in quartiles.index}

    # Define positions for annotations
    text_y_above = 1.1  # Above the box plot for 5th, 50th, 95th
    text_y_below = 0.7  # Below the box plot for 17th, 83rd

    for name in ["5q_dH_dT", "50q_dH_dT", "95q_dH_dT"]:
        value = quartiles[name]
        ax.text(
            value,
            text_y_above,
            f"{name.split('q')[0]}: {value:.1f} mm\n(~{iphones[name]:.1f} iPhones)",
            horizontalalignment="center",
            color=emulator_colors[emulator],
            fontweight="bold",
        )

    for name in ["17q_dH_dT", "83q_dH_dT"]:
        value = quartiles[name]
        ax.text(
            value,
            text_y_below,
            f"{name.split('q')[0]}: {value:.1f} mm\n(~{iphones[name]:.1f} iPhones)",
            horizontalalignment="center",
            color=emulator_colors[emulator],
            fontweight="bold",
        )

    ax.set_xlabel("Sea Level Rise (mm)")
    ax.set_yticks([])
    plt.grid(axis="x", linestyle="--", alpha=0.5)

    return fig


def main():
    st.title("Florida Sea Level Rise Projection")

    year = year_ui()

    # Google Drive File IDs for each shapefile component
    file_ids = {
        "ne_10m_coastline.shp": "13FSAfF40llhCUxxG9umplGU_WPJmxEn-",
        "ne_10m_coastline.dbf": "1Fx3ME7uAHux4hs8G4caw0M8M4iC-11yj",
        "ne_10m_coastline.prj": "1LudzsqtdTdzp29LYFwBxUI6gvpFwPBJS",
        "ne_10m_coastline.shx": "1GHbnb7RqGBcXNvd90APDGa-43wMAbwZX",
    }

    # Directory to save the shapefile components
    shp_folder = "data/"
    os.makedirs(shp_folder, exist_ok=True)

    # Download each required shapefile component
    for filename, file_id in file_ids.items():
        file_path = os.path.join(shp_folder, filename)
        if not os.path.exists(file_path):  # Avoid re-downloading
            print(f"Downloading {filename}...")
            gdown.download(
                f"https://drive.google.com/uc?id={file_id}", file_path, quiet=False
            )

    # Get emissions inputs & selected emulators
    co2 = emissions_ui()

    selected_emulator, emulator_colors = emulator_ui()

    # Create a Mapbox map centered on Florida
    fig = px.scatter_mapbox(
        lat=[27.9944024],  # Central latitude of Florida
        lon=[-81.7602544],  # Central longitude of Florida
        zoom=4.5,  # Zoom level to fit Florida
        mapbox_style="carto-positron",  # Clean Mapbox style
    )

    shapefile_path = os.path.join(shp_folder, "ne_10m_coastline.shp")
    coastline = gpd.read_file(shapefile_path)
    coastline = coastline[coastline["featurecla"] == "Coastline"]

    florida_bounds = {
        "lon_min": -84.6,  # Westernmost point (Pensacola)
        "lon_max": -80.0,  # Easternmost point (Atlantic Coast)
        "lat_min": 24.5,  # Southernmost point (Key West)
        "lat_max": 31.0,  # Northernmost point (Georgia border)
    }

    florida_coast = coastline.cx[
        florida_bounds["lon_min"] : florida_bounds["lon_max"],
        florida_bounds["lat_min"] : florida_bounds["lat_max"],
    ]

    coast_points = florida_coast.explode(
        index_parts=True
    )  # Convert lines to separate points
    coast_points = coast_points.geometry.apply(
        lambda geom: list(geom.coords) if geom.geom_type == "LineString" else None
    )
    coast_points = coast_points.explode().dropna().reset_index(drop=True)

    df_coastal = pd.DataFrame(coast_points.tolist(), columns=["Longitude", "Latitude"])

    if "Gaussian Process" == selected_emulator:
        path = f"data/GP_245/GP_Carbon_{co2}_Preds.csv"
        gp_df = pd.read_csv(path)
        gp_quartiles = gp_df[gp_df["year"] == year].iloc[0, 1:]
        gp_trace = px.scatter_mapbox(
            df_coastal,
            lat="Latitude",
            lon="Longitude",
            color_discrete_sequence=[emulator_colors["Gaussian Process"]],
            hover_data={
                "Latitude": False,
                "Longitude": False,
                "GP Sea Level Rise (mm)": np.round(
                    [gp_quartiles["50q_dH_dT"]] * len(df_coastal), 2
                ),
            },
        ).data[0]
        fig.add_trace(gp_trace)

        st.subheader("Sea Level Rise Projection with Uncertainty")
        st.write(
            "Change the Year or CO2 slider to reveal the median sea level rise (mm)."
        )
        line_plot(gp_df, year)

        st.subheader(f"GP Projected Sea Level Rise in {year}")
        box = plot_horizontal_boxplot(gp_quartiles, "Gaussian Process")
        st.pyplot(box)

    if "Random Forest" == selected_emulator:
        path = f"data/RF_245/RF_Carbon_{co2}_Preds.csv"
        rf_df = pd.read_csv(path)
        rf_quartiles = rf_df[rf_df["year"] == year].iloc[0, 1:]
        rf_trace = px.scatter_mapbox(
            df_coastal,
            lat="Latitude",
            lon="Longitude",
            color_discrete_sequence=[emulator_colors["Random Forest"]],
            hover_data={
                "Latitude": False,
                "Longitude": False,
                "RF Sea Level Rise (mm)": np.round(
                    [rf_quartiles["50q_dH_dT"]] * len(df_coastal), 2
                ),
            },
        ).data[0]
        fig.add_trace(rf_trace)

        st.subheader("Sea Level Rise Projection with Uncertainty")
        st.write(
            "Change the Year or CO2 slider to reveal the median sea level rise (mm)."
        )
        line_plot(rf_df, year)

        st.subheader(f"RF Projected Sea Level Rise in {year}")
        box = plot_horizontal_boxplot(rf_quartiles, "Random Forest")
        st.pyplot(box)

    st.subheader("Projected Sea Level Rise for Florida Under SSP245")
    st.write(f"Selected Year: {year}")
    st.plotly_chart(fig)
    # st.snow()


if __name__ == "__main__":
    main()
