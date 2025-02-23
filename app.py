import streamlit as st
import numpy as np
import plotly.express as px
import os
import pandas as pd
import gdown
import geopandas as gpd
import plotly.graph_objects as go
import matplotlib.pyplot as plt

max_co2 = 9500


def emissions_ui():
    st.sidebar.markdown("# Emissions 🌫️")
    # Change this to start at 0 and end at 9500
    co2 = st.sidebar.slider("Cumulative CO2 Amount (GtCO2)", 0, max_co2, 3340, 10)
    return co2


emulator_colors = {
    "Pattern Scaling": "#d55e00",
    "CNN-LTSM": "#cc79a7", 
    "Random Forest": "#009e73", 
    "Gaussian Process": "#0072b2", 
}

def emulator_ui():
    st.sidebar.markdown("# Select Emulator 👇")


    selected_emulator = st.sidebar.radio(
        "Choose an emulator:",
        list(emulator_colors.keys()),
        index=3  # Default selection
    )

    # Show selected emulator in a color-coded box
    color = emulator_colors[selected_emulator]

    return selected_emulator, {selected_emulator: color}


def line_plot(df, year):
    # Mask for showing past vs. future data
    mask_past = df[df["year"] <= year]

    # Create the plot
    fig = go.Figure()

    # --- Plot Median Line for Past ---
    fig.add_trace(go.Scatter(
        x=mask_past["year"], y=mask_past['50q_dH_dT'],
        mode='lines', name="Median Projection",
        line=dict(color="blue", width=2)
    ))

    # --- Plot Uncertainty Bands (5th-95th and 17th-83rd) for Past ---
    fig.add_trace(go.Scatter(
        x=np.concatenate([mask_past["year"], mask_past["year"][::-1]]),
        y=np.concatenate([mask_past["95q_dH_dT"], mask_past["5q_dH_dT"][::-1]]),
        fill='toself', fillcolor='rgba(0, 0, 255, 0.2)', 
        line=dict(color='rgba(255,255,255,0)'),
        name="90% Uncertainty (5th-95th)",
        hoverinfo="skip"
    ))

    fig.add_trace(go.Scatter(
        x=np.concatenate([mask_past["year"], mask_past["year"][::-1]]),
        y=np.concatenate([mask_past["83q_dH_dT"], mask_past["17q_dH_dT"][::-1]]),
        fill='toself', fillcolor='rgba(0, 0, 255, 0.3)', 
        line=dict(color='rgba(255,255,255,0)'),
        name="66% Uncertainty (17th-83rd)",
        hoverinfo="skip"
    ))

    fig.update_layout(
        xaxis=dict(
            title="Year",  # X-axis title
            title_font=dict(color="black"),  # X-axis title color
            tickfont=dict(color="black"),  # X-axis tick labels color
            range=[2015, 2100]
        ),
        yaxis=dict(
            title="Sea Level (mm)",  # Y-axis title
            title_font=dict(color="black"),  # Y-axis title color
            tickfont=dict(color="black"),  # Y-axis tick labels color
            range=[df["5q_dH_dT"].min() - 5, df["95q_dH_dT"].max() + 5]
        ),
        showlegend=True,
        legend=dict(
                font=dict(color="black"),  # Legend text color
                # bgcolor="white",            # Legend background color
                # bordercolor="white",        # Border color
                # borderwidth=2               # Border thickness
            ),
        paper_bgcolor="white",  # Outer background (outside the graph)
        plot_bgcolor="white",       # Inner plot background (inside the graph)
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
        quartiles['5q_dH_dT'], 
        quartiles['17q_dH_dT'], 
        quartiles['50q_dH_dT'], 
        quartiles['83q_dH_dT'], 
        quartiles['95q_dH_dT']
    ]

    ax.boxplot(percentiles, vert=False, patch_artist=True, 
               boxprops=dict(facecolor='lightgrey', color=emulator_colors[emulator]),
               medianprops=dict(color=emulator_colors[emulator], linewidth=2),
               whiskerprops=dict(color=emulator_colors[emulator], linewidth=1),
               capprops=dict(color=emulator_colors[emulator], linewidth=1),
               flierprops=dict(marker='o', color=emulator_colors[emulator], alpha=0.5))

    # Convert mm to iPhone thickness equivalents
    iphones = {name: quartiles[name] / 146.6 for name in quartiles.index}

    # Define positions for annotations
    text_y_above = 1.1  # Above the box plot for 5th, 50th, 95th
    text_y_below = 0.7  # Below the box plot for 17th, 83rd

    for name in ['5q_dH_dT', '50q_dH_dT', '95q_dH_dT']:
        value = quartiles[name]
        ax.text(value, text_y_above, f"{name.split('q')[0]}: {value:.1f} mm\n(~{iphones[name]:.1f} iPhones)", 
                horizontalalignment='center', color=emulator_colors[emulator], fontweight='bold')

    for name in ['17q_dH_dT', '83q_dH_dT']:
        value = quartiles[name]
        ax.text(value, text_y_below, f"{name.split('q')[0]}: {value:.1f} mm\n(~{iphones[name]:.1f} iPhones)", 
                horizontalalignment='center', color=emulator_colors[emulator], fontweight='bold')

    ax.set_xlabel("Sea Level Rise (mm)")
    ax.set_yticks([])
    plt.grid(axis="x", linestyle="--", alpha=0.5)

    return fig


def main():
    st.title("🌊 SeeRise: Visualizing Emulated Sea Level Rise on Florida")

    co2 = emissions_ui()

    selected_emulator, emulator_colors = emulator_ui()

    st.subheader(f"Selected emulator: {selected_emulator}")

    st.write("👋 Hello there! Welcome to our application that predicts sea level rise!")

    st.write("This application predicts the sea level rise under the assumption of SSP 245's values in the present year (2025). The only variable you are controlling is cumulative carbon dioxide. Your starting point is how many gigatons of carbon dioxide there is in 2025 (3340 giga tons).")

    with st.expander("🗣️ Click to learn about SSP 245"):
        st.write("SSP 245 stands for Shared Socioeconomic Pathway 2 with 4.5 W/m² Radiative Forcing by 2100.")
        st.write("Let's break down what that means...")
        st.write("Shared Socioeconomic Pathway 2 represents moderate global trends in population growth, economic development, and technology. So it is neither extreme sustainability nor extreme fossil-fuel reliance. There would be continued economic and population growth, but with persistent inequalities. There is also a mix of renewable and fossil-fuel-based energy sources.")
        st.write("We believe this is the most probable future. It's common name is the Middle of the Road Scenario.")
        st.write("RCP4.5 or 4.5 W/m² Radiative Forcing by 2100 means A stabilization scenario where emissions peak around mid-century and then decline. It assumes that mitigation policies slow global warming but do not fully stop it. This will lead to a moderate level of warming (~2.5-3°C above pre-industrial levels by 2100).")
        st.write("This allows the temperature to rise a little more that what the Paris Agreement hopes for.")

    st.write("You can select different emulators based off of the ClimateBench 🌎. These emulators use your input of cumulative carbon dioxide and the SSP 245's values for other greenhouse gases in 2025 to predict the temperature in 2100 🌡️. From there we predicted the sea level rise using linear regression.")

    with st.expander("Learn about your emulator:"):
        if selected_emulator == "Pattern Scaling":
            st.write('The pattern scaling model is the simplest emulator at our disposal. The model consists of many linear regression models trained on global mean temperature in different emission scenarios. These models regress desired variables (precipitation, diurnal temperature range, etc.) on global mean temperature which is the \"scaling\" element of the model. Once trained, the model takes a vector of global mean temperatures from a particular emission scenario, and predicts the desired variables using the inputs. This model is powerful yet simple because it can predict local values of particular variables using only globally averaged inputs.')
        elif selected_emulator == "Gaussian Process":
            st.write("A Gaussian Process (GP) model, used in the Climate Bench paper, is a probabilistic framework ideal for regression and classification tasks. GPs model functions by defining a prior characterized by a mean function, m(x), representing the expected value at x, and a covariance function, k(x,x'), which measures similarity between inputs x and x'. Using Bayesian inference, GPs update this prior with training data to produce a posterior distribution. For new inputs, predictions are made as a distribution with a mean (most likely value) and variance (uncertainty estimate).")
            st.write("GP models are well-suited for climate prediction. Climate systems are governed by complex, smooth, and often nonlinear relationships, which GPs can model through appropriately chosen kernels. Moreover, their ability to provide uncertainty estimates is invaluable when working with limited or noisy climate data, as these estimates can highlight regions where the model is less confident in its predictions. Finally, the interpretability of GP models aligns well with scientific practices, allowing researchers to explore the relationships captured by the covariance function and gain insights into the modeled climate dynamics.")
        elif selected_emulator == "Random Forest":
            st.write("Random Forest is an ensemble method that aggregates the predictions of multiple decision trees to enhance predictive performance. Decision trees, as the base models, are particularly effective at capturing non-linear relationships and interactions between variables but are prone to overfitting. Random Forest addresses this limitation by averaging the predictions of all individual trees, which reduces variance and increases robustness. This makes it well-suited for climate model emulation, where separate models are often developed for multiple target variables.")
            st.write("One key advantage of Random Forest in climate model emulation is its interpretability, which aids in informing decision-making. While a common drawback of Random Forest is its inability to extrapolate beyond the range of training data, this is not a significant concern in this context. Relevant predictions in climate modeling typically lie within the range defined by historical climate data and plausible scenarios, such as the low-emissions SSP126 and high-emissions SSP585 pathways. This makes Random Forest an effective and practical choice for emulating climate models.")
        else:
            st.write("A CNN-LSTM model combines Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) networks to capture spatial and temporal patterns in data. CNNs extract spatial features from input data, while LSTMs process sequential dependencies, making this architecture ideal for spatiotemporal modeling. By leveraging both components, CNN-LSTMs can learn complex relationships in time-series data while preserving spatial structures.")  
            st.write("CNN-LSTM models are particularly useful for climate model emulation. Climate data involves intricate spatial patterns and long-term temporal dependencies, which CNN-LSTMs effectively capture. They can emulate computationally expensive climate simulations by learning from historical climate outputs, enabling faster predictions. This approach is valuable for studying climate variability, extreme events, and future projections while reducing computational costs compared to full-scale climate models.")  


    year = 2025

    # Google Drive File IDs for each shapefile component
    file_ids = {
        "ne_10m_coastline.shp": "13FSAfF40llhCUxxG9umplGU_WPJmxEn-",
        "ne_10m_coastline.dbf": "1Fx3ME7uAHux4hs8G4caw0M8M4iC-11yj",
        "ne_10m_coastline.prj": "1LudzsqtdTdzp29LYFwBxUI6gvpFwPBJS",
        "ne_10m_coastline.shx": "1GHbnb7RqGBcXNvd90APDGa-43wMAbwZX"
    }

    # Directory to save the shapefile components
    shp_folder = "data/"
    os.makedirs(shp_folder, exist_ok=True)

    # Download each required shapefile component
    for filename, file_id in file_ids.items():
        file_path = os.path.join(shp_folder, filename)
        if not os.path.exists(file_path):  # Avoid re-downloading
            print(f"Downloading {filename}...")
            gdown.download(f"https://drive.google.com/uc?id={file_id}", file_path, quiet=False)

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
        "lat_min": 24.5,   # Southernmost point (Key West)
        "lat_max": 31.0,   # Northernmost point (Georgia border)
    }

    florida_coast = coastline.cx[
        florida_bounds["lon_min"]:florida_bounds["lon_max"], 
        florida_bounds["lat_min"]:florida_bounds["lat_max"]
    ]

    coast_points = florida_coast.explode(index_parts=True)  # Convert lines to separate points
    coast_points = coast_points.geometry.apply(lambda geom: list(geom.coords) if geom.geom_type == "LineString" else None)
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
            hover_data={"Latitude": False, "Longitude": False, "GP Sea Level Rise (mm)":
                         np.round([gp_quartiles['50q_dH_dT']]*len(df_coastal), 2)}
        ).data[0]
        fig.add_trace(gp_trace)

        # st.subheader("Sea Level Rise Projection with Uncertainty")
        # st.write("Change the Year or CO2 slider to reveal the median sea level rise (mm).")
        # line_plot(gp_df, year)

        st.subheader(f"GP Projected Sea Level Rise")
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

        # st.subheader("Sea Level Rise Projection with Uncertainty")
        # st.write(
        #     "Change the Year or CO2 slider to reveal the median sea level rise (mm)."
        # )
        # line_plot(rf_df, year)

        st.subheader(f"RF Projected Sea Level Rise")
        box = plot_horizontal_boxplot(rf_quartiles, "Random Forest")
        st.pyplot(box)

    if "CNN-LTSM" == selected_emulator:
        path = f"data/CNN_245/CNN_Carbon_{co2}_Preds.csv"
        cnn_df = pd.read_csv(path)
        cnn_quartiles = cnn_df[cnn_df["year"] == year].iloc[0, 1:]
        cnn_trace = px.scatter_mapbox(
            df_coastal,
            lat="Latitude",
            lon="Longitude",
            color_discrete_sequence=[emulator_colors["CNN-LTSM"]],
            hover_data={
                "Latitude": False,
                "Longitude": False,
                "RF Sea Level Rise (mm)": np.round(
                    [cnn_quartiles["50q_dH_dT"]] * len(df_coastal), 2
                ),
            },
        ).data[0]
        fig.add_trace(cnn_trace)

        # st.subheader("Sea Level Rise Projection with Uncertainty")
        # st.write(
        #     "Change the CO2 slider to reveal the median sea level rise (mm)."
        # )
        # line_plot(cnn_df, year)

        st.subheader(f"CNN Projected Sea Level Rise")
        box = plot_horizontal_boxplot(cnn_quartiles, "CNN-LTSM")
        st.pyplot(box)

    st.write("For your convenience we have determined an iPhone 📱 is 146.6mm. Now you can better visualize the sea level rise.")

    st.write("The figure above shows you a box plot of our sea level rise. The median (50th percentile) is a reasonable estimate of sea level rise.")

    st.write("😱 \"Wow, that's scary!\" However, even more concerning might be the following observation: land slopes. This means the sea level rise will flow inland, reducing our coastal lines.")

    st.subheader("Projected Sea Level Rise for Florida")
    st.plotly_chart(fig)
    st.write("Above, you will find another interactive figure. It looks at the elevation of the coastline of Florida. From there you can see how far the sea level will rise.")
    # st.snow()

if __name__ == "__main__":
    main()