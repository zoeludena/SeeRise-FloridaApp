import streamlit as st
import numpy as np
import tensorflow as tf
import xarray as xr
import plotly.express as px
from sklearn.linear_model import LinearRegression
import os
import urllib
import tarfile
import time
import pandas as pd

max_co2 = 9500.
max_ch4 = 0.8
max_so2 = 90.
max_bc = 9.

# File to download
DATA = {
    "climate_historical_data.tar.gz": {
        "url": "https://raw.githubusercontent.com/zoeludena/SeeRise-Florida/main/data/climate_historical_data.tar.gz",
        "description": "Historical Climate Data (1901-2014)",
        "extract": True,
    },
}

import streamlit as st
import time
import urllib.request
import tarfile
import os

def download_and_extract():
    for filename, info in DATA.items():
        if not os.path.exists(filename):
            urllib.request.urlretrieve(info["url"], filename)

        # Extract if needed
        if info.get("extract", False):
            with tarfile.open(filename, "r:gz") as tar:
                tar.extractall()

def load_historical_data():
    download_and_extract()  # Ensure files are downloaded before loading

    # Load historical climate data
    X_hist = np.load("X_hist.npy")  # Shape: (Time Steps, 1)
    y = np.load("y.npy")  # Shape: (Time Steps, 1)
    
    return X_hist, y

@st.cache_resource
def train_linear_regression():
    X_hist, y = load_historical_data()  # Load historical data
    hist_model = LinearRegression()
    hist_model.fit(X_hist, y)  # Train model
    return hist_model

# Sidebar for emissions input
def emissions_ui():
    st.sidebar.markdown("# Emissions")
    co2 = st.sidebar.slider("Cumulative CO2 Amount (GtCO2) in 2100", 0.0, max_co2, 1800., 10.)
    return co2

def emulator_ui():
    st.sidebar.markdown("# Select Emulator")

    # Define fixed colors for each emulator
    emulator_colors = {
        "Pattern Scaling": "#0072b2",
        "CNN-LTSM": "#d55e00", 
        "Random Forest": "#cc79a7", 
        "Gaussian Process": "#009e73", 
    }

    # Sidebar multiselect for emulator selection
    selected_emulators = st.sidebar.multiselect(
        "Choose one or more emulators:",
        list(emulator_colors.keys()),
        default=list(emulator_colors.keys())[0]
    )
    st.sidebar.markdown("### Emulator Color")

    # Show color-coded selections below
    for emulator in selected_emulators:
        color = emulator_colors[emulator]
        st.sidebar.markdown(
            f'<div style="background-color:{color}; padding:5px; border-radius:5px; color:white; text-align:center; margin-bottom:5px;">{emulator}</div>',
            unsafe_allow_html=True
        )

    return selected_emulators, {emulator: emulator_colors[emulator] for emulator in selected_emulators}



# Main app function
def main():
    st.title("Florida Sea Level Rise Projection")

    # Get emissions inputs & selected emulators
    co2 = emissions_ui()
    selected_emulators, emulator_colors = emulator_ui()

    st.subheader("Projected Sea Level Rise for Florida in 2100")

    # Train the Linear Regression model (Pattern Scaling)
    hist_model = train_linear_regression()

    # Create a Mapbox map centered on Florida
    fig = px.scatter_mapbox(
        lat=[27.9944024],  # Central latitude of Florida
        lon=[-81.7602544],  # Central longitude of Florida
        zoom=5,  # Zoom level to fit Florida
        mapbox_style="carto-positron",  # Clean Mapbox style
    )

    # Define Florida coastal locations (approximate lat/lon)
    coastal_locations = {
        "Miami": (25.7617, -80.1918),
        "Fort Lauderdale": (26.1224, -80.1373),
        "West Palm Beach": (26.7153, -80.0534),
        "Naples": (26.1420, -81.7948),
        "Tampa": (27.9506, -82.4572),
        "Sarasota": (27.3364, -82.5307),
        "Fort Myers": (26.6406, -81.8723),
        "St. Petersburg": (27.7676, -82.6403),
        "Daytona Beach": (29.2108, -81.0228),
        "Jacksonville": (30.3322, -81.6557),
        "Pensacola": (30.4213, -87.2169),
        "Key West": (24.5551, -81.7800)
    }

    # Convert to DataFrame
    df_coastal = pd.DataFrame(coastal_locations).T.reset_index()
    df_coastal.columns = ["City", "Latitude", "Longitude"]

    # Predict sea level rise for each coastal city
    df_coastal["Sea Level Rise (m)"] = np.round(hist_model.predict(np.array(co2).reshape(-1, 1))[-1][0]/1000, 2) # Make it meters

    # If "Pattern Scaling" is selected, add it to the map
    if "Pattern Scaling" in selected_emulators:
        # Add coastal cities with predicted sea level rise to the map
        fig.add_trace(px.scatter_mapbox(
            df_coastal,
            lat="Latitude",
            lon="Longitude",
            color_discrete_sequence=[emulator_colors["Pattern Scaling"]],  # Use predefined color
            hover_data={"Latitude": False, "Longitude": False, "Sea Level Rise (m)": True}
        ).data[0])

    # Display the map in Streamlit
    st.plotly_chart(fig)


# # Emulators to download
# EMULATORS = {
#     "ClimateBench_cnn_outputs_ssp245_predict_tas.nc": {
#         "url": "https://raw.githubusercontent.com/zoeludena/SeeRise-Florida/main/data/ClimateBench_cnn_outputs_ssp245_predict_tas.nc",
#         "description": "CNN-LTSM Emulator",
#         "extract": False,  # NetCDF files should NOT be extracted
#     },
#     "ClimateBench_gp_outputs_ssp245_predict_tas.tar.gz": {
#         "url": "https://raw.githubusercontent.com/zoeludena/SeeRise-Florida/main/data/ClimateBench_gp_outputs_ssp245_predict_tas.tar.gz",
#         "description": "Gaussian Process Emulator",
#         "extract": True,
#     },
#     "tuned_rf_outputs_ssp245_predict_tas.tar.gz": {
#         "url": "https://raw.githubusercontent.com/zoeludena/SeeRise-Florida/main/data/tuned_rf_outputs_ssp245_predict_tas.tar.gz",
#         "description": "Random Forest Emulator",
#         "extract": True,
#     },
# }

if __name__ == "__main__":
    main()
