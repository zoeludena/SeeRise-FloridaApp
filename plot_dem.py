import streamlit as st
import rasterio
from rasterio.windows import Window
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.colors as mcolors

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def plot_dem(sea_level, emulator, dem, crop_window=None, uploaded_file = None):
    """
    Plots the DEM of a location with an overlay showing projected flooding
    for a given sea level rise scenario. Flooded areas are shown in dark blue.
    """
    if dem == "Your Choice":
        if uploaded_file is None:
            st.warning("Please upload a DEM file before proceeding.")
            return  # Exit the function early to avoid errors

        # Process uploaded file
        with rasterio.MemoryFile(uploaded_file) as memfile:
            with memfile.open() as src:
                dem_array = src.read(1)
                profile = src.profile
                crs = src.crs
                bounds = src.bounds
    else:
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
        elif dem == "Titusville":
            DEM_PATH = os.path.join(BASE_DIR, "Titusville", "Titusville.dem")
        elif dem == "Cape Canaveral":
            DEM_PATH = os.path.join(BASE_DIR, "Cape_Canaveral", "Cape_Canaveral.dem")
        elif dem == "Audubon/Merritt Island":
            DEM_PATH = os.path.join(BASE_DIR, "Courtenay", "Courtenay.dem")
        elif dem == "Cedar Key":
            DEM_PATH = os.path.join(BASE_DIR, "Cedar_Key", "Cedar_Key.dem")
        elif dem == "Everglades City":
            DEM_PATH = os.path.join(BASE_DIR, "Everglades_City", "Everglades_City.dem")
        elif dem == "Naples":
            DEM_PATH = os.path.join(BASE_DIR, "Napels_North", "Napels_North.dem")
        
        with rasterio.open(DEM_PATH) as src:
            dem_array = src.read(1)
            profile = src.profile
            crs = src.crs
            bounds = src.bounds

            if crop_window:
                row_start, row_end, col_start, col_end = crop_window
                window = Window(col_start, row_start, col_end - col_start, row_end - row_start)
                dem_array = src.read(1, window=window)
            else:
                dem_array = src.read(1) 

    norm = mcolors.Normalize(vmin=0, vmax=5)

    # Replace NoData values with NaN.
    dem_array = np.where(dem_array == src.nodata, np.nan, dem_array)
    masked_dem = np.where((dem_array >= 0) & (dem_array <= 5), dem_array, np.nan)

    if sea_level > np.nanmax(dem_array):
        st.warning("Sea level is above the highest elevation in the DEM. Entire area would be flooded.")
    elif sea_level < np.nanmin(dem_array):
        st.warning("Sea level is below the lowest elevation in the DEM. No flooding expected.")

    # Create a mask for flooded areas and correct for historical sea level rise.
    slr_correction = 0.4637622 # Calculated using historical satellite data.
    flooded_mask = dem_array <= (sea_level + slr_correction)

    # Plot DEM using the terrain colormap
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(masked_dem, cmap="grey", origin="upper") 
    plt.colorbar(cax, label="Elevation (m)")

    # Overlay the flooded areas in dark blue using a contour plot
    if np.any(flooded_mask):  # Ensure we have flooded areas before plotting 
        ax.contourf(dem_array, levels=[np.nanmin(dem_array), sea_level], colors=["dodgerblue"], alpha=0.6)

    ax.set_title(f"{emulator} DEM with {sea_level:.2f}m Sea Level Rise")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    if dem not in ["Sanibel Island", "Fort Myers Beach", "Everglades City"]:
        if dem == "Audubon/Merritt Island":
            ax.text(
                0.5, -0.20, "*Note: Areas above 5m are also white.",
                fontsize=10, ha="right", transform=ax.transAxes, color="black"
            )
        else:
            ax.text(
                0.5, -0.12, "*Note: Areas above 5m are also white.",
                fontsize=10, ha="right", transform=ax.transAxes, color="black"
            )

    st.pyplot(fig)

    # # Display metadata
    # st.write(f"**Array Shape:** {dem_array.shape}")
    # st.write(f"**CRS:** {crs}")
    # st.write(f"**Bounds:** {bounds}")
    # st.write(f"**Min Elevation:** {np.nanmin(dem_array):.2f} m")
    # st.write(f"**Max Elevation:** {np.nanmax(dem_array):.2f} m")
