import streamlit as st
import numpy as np
import tensorflow as tf
import xarray as xr
import plotly.express as px
from sklearn.linear_model import LinearRegression

# File to download
DATA = {
    "climate_historical_data.tar.gz": {
        "url": "https://raw.githubusercontent.com/zoeludena/SeeRise-Florida/main/data/climate_historical_data.tar.gz",
        "description": "Historical Climate Data (1901-2014)",
        "extract": True,
    },
}

def download_and_extract():
    for filename, info in DATA.items():
        if not os.path.exists(filename):
            st.sidebar.warning(f"Downloading {info['description']}...")
            urllib.request.urlretrieve(info["url"], filename)
            st.sidebar.success(f"Downloaded {info['description']}!")

        # Extract if needed
        if info.get("extract", False):
            with tarfile.open(filename, "r:gz") as tar:
                tar.extractall()
                st.sidebar.success(f"Extracted {info['description']}!")

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
    co2 = st.sidebar.slider("CO2 concentrations (GtCO2)", 0.0, max_co2, 1800., 10.)
    return [co2, 0.3, 85., 7.]  # Default values for CH4, SO2, and BC are kept as placeholders

# Sidebar for emulator selection
def emulator_ui():
    st.sidebar.markdown("# Select Emulator")
    available_emulators = ["Pattern Scaling", "CNN-LTSM", "Random Forest", "Gaussian Process"]
    selected_emulators = st.sidebar.multiselect("Choose one or more emulators", available_emulators, default=[available_emulators[0]])
    return selected_emulators

# Main app function
def main():
    st.title("Florida Sea Level Rise Projection")
    # co2, ch4, so2, bc = emissions_ui()
    selected_emulators = emulator_ui()
    
    st.subheader("Projected Sea Level Rise for Florida")
    
    # Define some major Florida locations
    data = {
        "name": ["Miami", "Orlando", "Tampa", "Tallahassee", "Jacksonville"],
        "latitude": [25.7617, 28.5383, 27.9506, 30.4383, 30.3322],
        "longitude": [-80.1918, -81.3792, -82.4572, -84.2807, -81.6557],
    }

    df = pd.DataFrame(data)

    # Create an interactive Mapbox map
    fig = px.scatter_mapbox(
        df,
        lat="latitude",
        lon="longitude",
        text="name",
        zoom=5,  # Adjust zoom level
        mapbox_style="carto-positron",  # Map style
        title="Major Cities in Florida"
    )

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
