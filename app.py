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
from esem import gp_model

max_co2 = 9500.
max_ch4 = 0.8
max_so2 = 90.
max_bc = 9.

normalize_co2 = lambda data: data / max_co2
normalize_ch4 = lambda data: data / max_ch4


# Define Google Drive file IDs
file_ids = {
    "inputs_historical.nc": "1z0OxBEBFNOknn1W6mGwNqQ9NocFCb9Q1",
    "inputs_ssp126.nc": "1KNA6VcuNaACNTFzJRyt3RwH2sDoZnTZG",
    "inputs_ssp245.nc": "1mgyUzx6m4Jl5Nmvzw7ejmsshhTg4x82T",
    "inputs_ssp370.nc": "1DgyGrMVLxKo7aVAJgHC1kk_wgFCoN4H7",
    "inputs_ssp585.nc": "1XgZ12hLxd-dP7_09jjaHqX4TxUA66hox",
    "outputs_historical.nc": "1QmDcFjWX4dohh4ZqW5Ga4GS5k9_iC8Zl",
    "outputs_ssp126.nc": "1QwRxX0ZcG4nEtMlwm-_km4mVgG2RCKqL",
    "outputs_ssp245.nc": "1-eCNMpVtDlHHX7AN3vhzTONOM7HuuFZ9",
    "outputs_ssp370.nc": "1sL4anpD9JnnqVV52vDSqE5gnOIY-HpXt",
    "outputs_ssp585.nc": "1SHZLAZdd3Mbyr0ZXBG5ODBzXXmJ9KLh1",
}

# Directory to save downloaded files
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

def download_nc_files():
    for filename, file_id in file_ids.items():
        file_path = os.path.join(data_dir, filename)

        if not os.path.exists(file_path):  # Check if file already exists
            url = f"https://drive.google.com/uc?id={file_id}"
            print(f"Downloading {filename}...")
            gdown.download(url, file_path, quiet=False)
        else:
            print(f"{filename} already exists, skipping...")


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

def gp_emulator(co2):
    download_nc_files()  # Ensure .nc files are downloaded

    data_path = "data/"  

    required_files = [
        'inputs_historical.nc', 'inputs_ssp585.nc',
        'inputs_ssp126.nc', 'inputs_ssp370.nc',
        'outputs_historical.nc', 'outputs_ssp585.nc',
        'outputs_ssp245.nc', 'inputs_ssp245.nc'
    ]

    missing_files = [f for f in required_files if not os.path.exists(os.path.join(data_path, f))]
    if missing_files:
        raise FileNotFoundError(f"Missing files: {missing_files}. Ensure all .nc files are downloaded.")

    # Load training data
    X = xr.open_mfdataset([
        os.path.join(data_path, 'inputs_historical.nc'),
        os.path.join(data_path, 'inputs_ssp585.nc'),
        os.path.join(data_path, 'inputs_ssp126.nc'),
        os.path.join(data_path, 'inputs_ssp370.nc')
    ], engine="netcdf4").compute()

    Y = xr.concat([
        xr.open_dataset(os.path.join(data_path, 'outputs_historical.nc'), engine="netcdf4").sel(member=2),
        xr.open_dataset(os.path.join(data_path, 'outputs_ssp585.nc'), engine="netcdf4").sel(member=1)
    ], dim='time').compute()

    # Ensure input & output data have the same number of samples
    min_samples = min(len(X["CO2"].data), len(Y["tas"].data))
    X = X.isel(time=slice(0, min_samples))  # Trim inputs
    Y = Y.isel(time=slice(0, min_samples))  # Trim outputs

    # EOF Analysis
    bc_solver = Eof(X['BC'])
    bc_pcs = bc_solver.pcs(npcs=5, pcscaling=1)
    so2_solver = Eof(X['SO2'])
    so2_pcs = so2_solver.pcs(npcs=5, pcscaling=1)

    bc_df = bc_pcs.to_dataframe().unstack('mode')
    bc_df.columns = [f"BC_{i}" for i in range(5)]
    so2_df = so2_pcs.to_dataframe().unstack('mode')
    so2_df.columns = [f"SO2_{i}" for i in range(5)]

    # **Modify input data to use selected CO₂ value**
    leading_historical_inputs = pd.DataFrame({
        "CO2": normalize_co2(np.full((min_samples,), co2)),  # Ensure correct shape
        "CH4": normalize_ch4(X["CH4"].data[:min_samples])  # Keep CH₄ constant
    }, index=X["CO2"].coords['time'].data[:min_samples])

    leading_historical_inputs = pd.concat([leading_historical_inputs, bc_df, so2_df], axis=1)

    # Train GP Model
    tas_gp = gp_model(leading_historical_inputs, Y["tas"])
    tas_gp.train()

    # Predict global mean temperature based on the selected CO₂ level
    m_tas, _ = tas_gp.predict(leading_historical_inputs)
    tas_global_mean = m_tas.mean(dim=("lat", "lon"))

    return tas_global_mean.values



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

    # TODO: Change to TAS then to CO2 -> For now just a placeholder
    # Predict sea level rise for each coastal city
    df_coastal["Sea Level Rise (m)"] = np.round(hist_model.predict(np.array(co2).reshape(-1, 1))[-1][0]/1000, 2) # Make it meters

    # WORK IN PROGRESS
    tas_global_mean = gp_emulator(co2).reshape(-1, 1)

    # GP Model
    # The output units of hist_model (Linear Regression model) depend on how it was trained. If hist_model was trained to predict sea level rise in meters (m) or millimeters (mm) using temperature as input, then its output will be in that unit.
    df_coastal["GP Sea Level Rise (mm)"] = np.round(hist_model.predict(tas_global_mean)[-1][0], 2) # Giving it Kelvin Temperature
        #Is it in mm??? I am so confused.

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

    if "Gaussian Process" in selected_emulators:
        fig.add_trace(px.scatter_mapbox(
            df_coastal,
            lat="Latitude",
            lon="Longitude",
            color_discrete_sequence=[emulator_colors["Gaussian Process"]],  # Use predefined color
            hover_data={"Latitude": False, "Longitude": False, "GP Sea Level Rise (mm)": True}
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
