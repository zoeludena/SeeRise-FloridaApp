import streamlit as st
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def plot_dem(sea_level, emulator, dem):
    """
    Plots the DEM of a location with an overlay showing projected flooding
    for a given sea level rise scenario. Flooded areas are shown in dark blue.
    """
    if dem == "Sanibel Island":
        DEM_PATH = os.path.join(BASE_DIR, "Sanibel", "Sanibel.dem")
    elif dem == "Miami":
        DEM_PATH = os.path.join(BASE_DIR, "Miami", "Miami.dem")
    elif dem == "Tampa":
        DEM_PATH = os.path.join(BASE_DIR, "Tampa", "Tampa.dem")
    elif dem == "Hobe Sound":
        DEM_PATH = os.path.join(BASE_DIR, "Hobe_Sound", "Hobe.dem")
    elif dem == "Fort Myers Beach":
        DEM_PATH = os.path.join(BASE_DIR, "Fort_Myers", "Fort_Myers.dem")
    
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
            ax.contourf(dem_array, levels=[np.nanmin(dem_array), sea_level], colors=["cornflowerblue"], alpha=0.6)

        ax.set_title(f"{emulator} {dem} DEM with {sea_level:.2f}m Sea Level Rise")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        st.pyplot(fig)

        # # Display metadata
        # st.write(f"**Array Shape:** {dem_array.shape}")
        # st.write(f"**CRS:** {crs}")
        # st.write(f"**Bounds:** {bounds}")
        # st.write(f"**Min Elevation:** {np.nanmin(dem_array):.2f} m")
        # st.write(f"**Max Elevation:** {np.nanmax(dem_array):.2f} m")
