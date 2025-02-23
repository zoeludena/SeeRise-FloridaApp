import streamlit as st
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # Get current directory
DEM_PATH = os.path.join(BASE_DIR, "Tampa", "Tampa.dem")  # Ensure correct casing

def plot_tampa_dem(sea_level):
    """
    Plots the DEM of Sanibel Island with an overlay showing projected flooding
    for a given sea level rise scenario. Flooded areas are shown in dark blue.
    """
    with rasterio.open(DEM_PATH) as src:
        dem_array = src.read(1)  # Read first band
        profile = src.profile
        crs = src.crs
        bounds = src.bounds

        # Replace NoData values with NaN
        dem_array = np.where(dem_array == src.nodata, np.nan, dem_array)

        # st.write(f"DEM min elevation: {np.nanmin(dem_array):.2f}m, max elevation: {np.nanmax(dem_array):.2f}m")
        # st.write(f"Selected sea level rise: {sea_level:.2f}m")

        if sea_level > np.nanmax(dem_array):
            st.warning("Sea level is above the highest elevation in the DEM. Entire area would be flooded.")
        elif sea_level < np.nanmin(dem_array):
            st.warning("Sea level is below the lowest elevation in the DEM. No flooding expected.")

        # Create a mask for flooded areas
        flooded_mask = dem_array <= sea_level

        # Plot DEM using the terrain colormap
        fig, ax = plt.subplots(figsize=(8, 6))
        cax = ax.imshow(dem_array, cmap="grey", origin="upper")
        plt.colorbar(cax, label="Elevation (m)")

        # Overlay the flooded areas in dark blue using a contour plot
        if np.any(flooded_mask):  # Ensure we have flooded areas before plotting
            ax.contourf(dem_array, levels=[np.nanmin(dem_array), sea_level], colors=["darkblue"], alpha=0.6)

        ax.set_title(f"Tampa DEM with {sea_level:.2f}m Sea Level Rise")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        st.pyplot(fig)

        # # Display metadata
        # st.write(f"**Array Shape:** {dem_array.shape}")
        # st.write(f"**CRS:** {crs}")
        # st.write(f"**Bounds:** {bounds}")
        # st.write(f"**Min Elevation:** {np.nanmin(dem_array):.2f} m")
        # st.write(f"**Max Elevation:** {np.nanmax(dem_array):.2f} m")