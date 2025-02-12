import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px

# Placeholder function for emulator models
def run_emulator(emulator, co2_concentration):
    if emulator == "Model A":
        temp_change = 0.02 * (co2_concentration - 280)
    elif emulator == "Model B":
        temp_change = 0.025 * (co2_concentration - 280)
    else:
        temp_change = 0.03 * (co2_concentration - 280)
    return temp_change

# Placeholder function for sea level rise calculation
def calculate_sea_level_rise(temp_change):
    return 0.4 * temp_change  # Example equation

st.title("Florida Sea Level Rise Predictor")

co2_concentration = st.slider("Select CO2 Concentration (ppm):", 280, 1000, 420)

emulator = st.selectbox("Choose an Emulator:", ["Model A", "Model B", "Model C"])

temp_change = run_emulator(emulator, co2_concentration)
sea_level_rise = calculate_sea_level_rise(temp_change)

st.write(f"Predicted Temperature Change: {temp_change:.2f}°C")
st.write(f"Predicted Sea Level Rise: {sea_level_rise:.2f} meters")

florida_map = px.scatter_mapbox(
    pd.DataFrame({"lat": [27.994402], "lon": [-81.760254], "Sea Level Rise (m)": [sea_level_rise]}),
    lat="lat",
    lon="lon",
    size="Sea Level Rise (m)",
    color="Sea Level Rise (m)",
    zoom=6,
    mapbox_style="carto-positron"
)
st.plotly_chart(florida_map)
