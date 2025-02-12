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

# Sidebar for emulator selection
def emulator_ui():
    st.sidebar.markdown("# Select Emulator")
    available_emulators = ["emulator", "alternate_emulator_1", "alternate_emulator_2"]
    selected_emulators = st.sidebar.multiselect("Choose one or more emulators", available_emulators, default=[available_emulators[0]])
    return selected_emulators

# Main app function
def main():
    st.title("Florida Sea Level Rise Projection")
    co2, ch4, so2, bc = emissions_ui()
    selected_emulators = emulator_ui()
    
    st.subheader("Projected Sea Level Rise for Florida")
    
    fig = px.scatter_mapbox(mapbox_style="carto-positron", zoom=5, title="Florida Sea Level Rise Projections")
    
    # Generate lat/lon grid for Florida (approximate bounds)
    lat_range = np.linspace(24, 31, 50)
    lon_range = np.linspace(-88, -79, 50)
    lat, lon = np.meshgrid(lat_range, lon_range)
    
    for emulator in selected_emulators:
        temperature = predict_temperature(emulator, co2, ch4, so2, bc)
        sea_level_rise = calculate_sea_level_rise(temperature)
        
        # Flatten for visualization
        df = xr.DataArray(sea_level_rise, coords={'latitude': lat.flatten(), 'longitude': lon.flatten()})
        df = df.to_dataframe().reset_index()
        df.rename(columns={0: f"Sea Level Rise ({emulator}) (m)"}, inplace=True)
        
        fig.add_trace(px.scatter_mapbox(df, lat="latitude", lon="longitude", color=df.columns[-1]).data[0])
    
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()
