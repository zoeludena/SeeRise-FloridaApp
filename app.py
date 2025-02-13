import streamlit as st
import numpy as np
import tensorflow as tf
import xarray as xr
import plotly.express as px

# Constants for normalization
max_co2 = 9500.
max_ch4 = 0.8
max_so2 = 90.
max_bc = 9.

# Function to normalize inputs
def normalize_inputs(data):
    return np.asarray(data) / np.asarray([max_co2, max_ch4, max_so2, max_so2])

def download_file(file_name):
    import os
    import urllib.request
    import tarfile

    if file_name not in EXTERNAL_DEPENDENCIES:
        st.error(f"File '{file_name}' not found in EXTERNAL_DEPENDENCIES.")
        return
    
    file_path = file_name  # Save the file in the same directory
    file_info = EXTERNAL_DEPENDENCIES[file_name]
    
    if os.path.exists(file_path):
        st.success(f"'{file_name}' already exists.")
        return
    
    # UI elements for progress display
    weights_warning = st.warning(f"Downloading {file_info['description']}...")
    progress_bar = st.progress(0)

    try:
        with open(file_path, "wb") as output_file:
            with urllib.request.urlopen(file_info["url"]) as response:
                length = int(response.info()["Content-Length"])
                counter = 0
                MEGABYTES = 2.0 ** 20.0

                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)
                    
                    # Update progress bar
                    weights_warning.warning(f"Downloading {file_info['description']} ({counter / MEGABYTES:.2f}/{length / MEGABYTES:.2f} MB)")
                    progress_bar.progress(min(counter / length, 1.0))

        st.success(f"Downloaded {file_info['description']} successfully.")

        # Extract tar.gz files if needed
        if file_info.get("extract", False) and file_path.endswith('.tar.gz'):
            with tarfile.open(file_path, "r:gz") as tar:
                tar.extractall()
                st.success(f"Extracted {file_info['description']}.")

    finally:
        weights_warning.empty()
        progress_bar.empty()


# Sea level rise calculation (example function)
def calculate_sea_level_rise(temp):
    return 3.0 * temp  # Placeholder linear relationship

# Load emulator model
@st.cache_resource
def load_model(model_path):
    return tf.saved_model.load(model_path)

# Run emulator to predict temperature
def predict_temperature(emulator, co2, ch4, so2, bc):
    model = load_model(emulator)
    inputs = tf.convert_to_tensor([[co2, ch4, so2, bc]], dtype=tf.float64)
    posterior_mean, _ = model.predict_f_compiled(inputs)
    return np.reshape(posterior_mean, [96, 144])

# Sidebar for emissions input
def emissions_ui():
    st.sidebar.markdown("# Emissions")
    co2 = st.sidebar.slider("CO2 concentrations (GtCO2)", 0.0, max_co2, 1800., 10.)
    return [co2, 0.3, 85., 7.]  # Default values for CH4, SO2, and BC are kept as placeholders

# Select emulators
def emulator_ui():
    st.sidebar.markdown("# Select Emulator")
    available_emulators = [key for key in EXTERNAL_DEPENDENCIES if "emulator" in key]
    
    selected_emulators = st.sidebar.multiselect(
        "Choose one or more emulators",
        available_emulators,
        default=[available_emulators[0]] if available_emulators else []
    )
    
    return selected_emulators


def main():
    st.title("Florida Sea Level Rise Projection")

    # Ensure required files are downloaded
    for file in EXTERNAL_DEPENDENCIES.keys():
        download_file(file)

    co2, ch4, so2, bc = emissions_ui()
    selected_emulators = emulator_ui()

    st.subheader("Projected Sea Level Rise for Florida")

    fig = px.scatter_mapbox(mapbox_style="carto-positron", zoom=5, title="Florida Sea Level Rise Projections")

    lat_range = np.linspace(24, 31, 50)
    lon_range = np.linspace(-88, -79, 50)
    lat, lon = np.meshgrid(lat_range, lon_range)

    for emulator in selected_emulators:
        temperature = predict_temperature(emulator, co2, ch4, so2, bc)
        sea_level_rise = calculate_sea_level_rise(temperature)

        df = xr.DataArray(sea_level_rise, coords={'latitude': lat.flatten(), 'longitude': lon.flatten()})
        df = df.to_dataframe().reset_index()
        df.rename(columns={0: f"Sea Level Rise ({emulator}) (m)"}, inplace=True)

        fig.add_trace(px.scatter_mapbox(df, lat="latitude", lon="longitude", color=df.columns[-1]).data[0])

    st.plotly_chart(fig)


# Files to download
EXTERNAL_DEPENDENCIES = {
    "emulator_model.tar.gz": {
        "url": "https://example.com/emulator_model.tar.gz",
        "description": "Main Emulator Model",
        "extract": True,  # Whether to extract the file after downloading
    },
    "alternate_emulator_1.tar.gz": {
        "url": "https://example.com/alternate_emulator_1.tar.gz",
        "description": "Alternative Emulator 1",
        "extract": True,
    },
    "climate_data.nc": {
        "url": "https://example.com/climate_data.nc",
        "description": "Climate Data NetCDF File",
        "extract": False,
    },
}


if __name__ == "__main__":
    main()
