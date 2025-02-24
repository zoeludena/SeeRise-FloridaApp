import streamlit as st
import numpy as np
import plotly.express as px
import os
import pandas as pd
import gdown
import geopandas as gpd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from plot_dem import plot_dem

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
        index=0,  # Default selection
    )

    # Show selected emulator in a color-coded box
    color = emulator_colors[selected_emulator]

    return selected_emulator, {selected_emulator: color}


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
        quartiles["5q_dH_dT"],
        quartiles["17q_dH_dT"],
        quartiles["50q_dH_dT"],
        quartiles["83q_dH_dT"],
        quartiles["95q_dH_dT"],
    ]

    ax.boxplot(
        percentiles,
        vert=False,
        patch_artist=True,
        boxprops=dict(facecolor="lightgrey", color=emulator_colors[emulator]),
        medianprops=dict(color=emulator_colors[emulator], linewidth=2),
        whiskerprops=dict(color=emulator_colors[emulator], linewidth=1),
        capprops=dict(color=emulator_colors[emulator], linewidth=1),
        flierprops=dict(marker="o", color=emulator_colors[emulator], alpha=0.5),
    )

    # ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # Convert mm to iPhone thickness equivalents
    iphones = {name: quartiles[name] / 146.6 for name in quartiles.index}

    # Define positions for annotations
    text_y_above = 1.1  # Above the box plot for 5th, 50th, 95th
    text_y_below = 0.7  # Below the box plot for 17th, 83rd

    for name in ["5q_dH_dT", "50q_dH_dT", "95q_dH_dT"]:
        value = quartiles[name]
        ax.text(
            value,
            text_y_above,
            f"{name.split('q')[0]}: {value:.1f} mm\n(~{iphones[name]:.1f} iPhones)",
            horizontalalignment="center",
            color=emulator_colors[emulator],
            fontweight="bold",
        )

    for name in ["17q_dH_dT", "83q_dH_dT"]:
        value = quartiles[name]
        ax.text(
            value,
            text_y_below,
            f"{name.split('q')[0]}: {value:.1f} mm\n(~{iphones[name]:.1f} iPhones)",
            horizontalalignment="center",
            color=emulator_colors[emulator],
            fontweight="bold",
        )

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

    st.write(
        "This application predicts the sea level rise under the assumption of SSP 245's values in 2100. The only variable you are controlling is cumulative carbon dioxide. Your starting point is how many gigatons of carbon dioxide there is in 2025 (3340 giga tons)."
    )

    with st.expander("📣 Click to learn about SSP 245"):
        st.write(
            "SSP 245 stands for Shared Socioeconomic Pathway 2 with 4.5 W/m² Radiative Forcing by 2100."
        )
        st.write("Let's break down what that means...")
        st.write(
            "Shared Socioeconomic Pathway 2 represents moderate global trends in population growth, economic development, and technology. So it is neither extreme sustainability nor extreme fossil-fuel reliance. There would be continued economic and population growth, but with persistent inequalities. There is also a mix of renewable and fossil-fuel-based energy sources."
        )
        st.write(
            "We believe this is the most probable future. It's common name is the Middle of the Road Scenario."
        )
        st.write(
            "RCP4.5 or 4.5 W/m² Radiative Forcing by 2100 means A stabilization scenario where emissions peak around mid-century and then decline. It assumes that mitigation policies slow global warming but do not fully stop it. This will lead to a moderate level of warming (~2.5-3°C above pre-industrial levels by 2100)."
        )
        st.write(
            "This allows the temperature to rise a little more that what the Paris Agreement hopes for."
        )

    st.write(
        "You can select different emulators based off of the ClimateBench 🌎. These emulators use your input of cumulative carbon dioxide and the SSP 245's values for other greenhouse gases in 2100 to predict the temperature 🌡️. From there we predicted the sea level rise using linear regression."
    )

    with st.expander("🤖 Learn about your emulator:"):
        if selected_emulator == "Pattern Scaling":
            st.write(
                'The pattern scaling model is the simplest emulator at our disposal. The model consists of many linear regression models trained on global mean temperature in different emission scenarios. These models regress desired variables (precipitation, diurnal temperature range, etc.) on global mean temperature which is the "scaling" element of the model. Once trained, the model takes a vector of global mean temperatures from a particular emission scenario, and predicts the desired variables using the inputs. This model is powerful yet simple because it can predict local values of particular variables using only globally averaged inputs.'
            )
        elif selected_emulator == "Gaussian Process":
            st.write(
                "A Gaussian Process (GP) model, used in the Climate Bench paper, is a probabilistic framework ideal for regression and classification tasks. GPs model functions by defining a prior characterized by a mean function, m(x), representing the expected value at x, and a covariance function, k(x,x'), which measures similarity between inputs x and x'. Using Bayesian inference, GPs update this prior with training data to produce a posterior distribution. For new inputs, predictions are made as a distribution with a mean (most likely value) and variance (uncertainty estimate)."
            )
            st.write(
                "GP models are well-suited for climate prediction. Climate systems are governed by complex, smooth, and often nonlinear relationships, which GPs can model through appropriately chosen kernels. Moreover, their ability to provide uncertainty estimates is invaluable when working with limited or noisy climate data, as these estimates can highlight regions where the model is less confident in its predictions. Finally, the interpretability of GP models aligns well with scientific practices, allowing researchers to explore the relationships captured by the covariance function and gain insights into the modeled climate dynamics."
            )
        elif selected_emulator == "Random Forest":
            st.write(
                "Random Forest is an ensemble method that aggregates the predictions of multiple decision trees to enhance predictive performance. Decision trees, as the base models, are particularly effective at capturing non-linear relationships and interactions between variables but are prone to overfitting. Random Forest addresses this limitation by averaging the predictions of all individual trees, which reduces variance and increases robustness. This makes it well-suited for climate model emulation, where separate models are often developed for multiple target variables."
            )
            st.write(
                "One key advantage of Random Forest in climate model emulation is its interpretability, which aids in informing decision-making. While a common drawback of Random Forest is its inability to extrapolate beyond the range of training data, this is not a significant concern in this context. Relevant predictions in climate modeling typically lie within the range defined by historical climate data and plausible scenarios, such as the low-emissions SSP126 and high-emissions SSP585 pathways. This makes Random Forest an effective and practical choice for emulating climate models."
            )
        else:
            st.write(
                "A CNN-LSTM model combines Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) networks to capture spatial and temporal patterns in data. CNNs extract spatial features from input data, while LSTMs process sequential dependencies. By leveraging both components, CNN-LSTMs can learn complex relationships in time-series data while preserving spatial structures."
            )
            st.write(
                "CNN-LSTM models are particularly useful for climate model emulation. Climate data involves intricate spatial patterns and long-term temporal dependencies, which CNN-LSTMs effectively capture. They can emulate computationally expensive climate simulations by learning from historical climate outputs, enabling faster predictions. This approach is valuable for studying climate variability, extreme events, and future projections while reducing computational costs compared to full-scale climate models."
            )

    year = 2100

    if "Pattern Scaling" == selected_emulator:
        path = f"data/PS_Carbon/PS_Carbon_{co2}_Preds.csv"
        ps_df = pd.read_csv(path)
        ps_quartiles = ps_df[ps_df["year"] == year].iloc[0, 1:]
        st.subheader(f"PS Projected Sea Level Rise")
        box = plot_horizontal_boxplot(ps_quartiles, "Pattern Scaling")
        st.pyplot(box)

        sea_level_rise = ps_quartiles["50q_dH_dT"] / 1000  # Convert mm to meters

    if "Gaussian Process" == selected_emulator:
        # path = f"data/GP_245/GP_Carbon_{co2}_Preds.csv"
        path = f"data/GP_245_Linear/GP_Carbon_{co2}_Preds.csv"
        gp_df = pd.read_csv(path)
        gp_quartiles = gp_df[gp_df["year"] == year].iloc[0, 1:]
        st.subheader(f"GP Projected Sea Level Rise")
        box = plot_horizontal_boxplot(gp_quartiles, "Gaussian Process")
        st.pyplot(box)

        sea_level_rise = gp_quartiles["50q_dH_dT"] / 1000  # Convert mm to meters

    if "Random Forest" == selected_emulator:
        path = f"data/RF_linear/RF_Carbon_{co2}_Preds.csv"
        rf_df = pd.read_csv(path)
        rf_quartiles = rf_df[rf_df["year"] == year].iloc[0, 1:]
        st.subheader(f"RF Projected Sea Level Rise")
        box = plot_horizontal_boxplot(rf_quartiles, "Random Forest")
        st.pyplot(box)

        sea_level_rise = rf_quartiles["50q_dH_dT"] / 1000  # Convert mm to meters

    if "CNN-LTSM" == selected_emulator:
        path = f"data/CNN_245_Linear/CNN_Carbon_{co2}_Preds.csv"
        cnn_df = pd.read_csv(path)
        cnn_quartiles = cnn_df[cnn_df["year"] == year].iloc[0, 1:]
        st.subheader(f"CNN Projected Sea Level Rise")
        box = plot_horizontal_boxplot(cnn_quartiles, "CNN-LTSM")
        st.pyplot(box)

        sea_level_rise = cnn_quartiles["50q_dH_dT"] / 1000  # Convert mm to meters

    st.write(
        "For your convenience we have determined an iPhone 📱 is 146.6mm. Now you can better visualize the sea level rise."
    )

    st.write(
        "The figure above shows you a box plot of our sea level rise. The median (50th percentile) is a reasonable estimate of sea level rise."
    )

    st.write(
        '😱 "Wow, that\'s scary!" However, even more concerning might be the following observation: land slopes. This means the sea level rise will flow inland, reducing our coastal lines.'
    )

    st.subheader("Sanibel Island Sea Level Rise in 2100")

    if selected_emulator == "Pattern Scaling":
        plot_dem(sea_level_rise, "Pattern Scaling", "Sanibel Island")
    if selected_emulator == "Gaussian Process":
        plot_dem(sea_level_rise, "Gaussian Process", "Sanibel Island")
    if "CNN-LTSM" == selected_emulator:
        plot_dem(sea_level_rise, "CNN-LTSM", "Sanibel Island")
    if "Random Forest" == selected_emulator:
        plot_dem(sea_level_rise, "Random Forest", "Sanibel Island")

    with st.expander("What is DEM? 🗺️"):
        st.write("DEM stands for digital elevation model. It aims to recreate the topographic surface of the Earth excluding buildings and foliage.")
        st.write("We wanted to show the digital representation of the contours of earths surface. This way it would be easier for our audience to visualize sea level rise.")
        st.write("We obtained these DEM files from the Land Boundary Information System (LABINS). Under mapping data.")

    st.write(
        "Above you can see Sanibel Island. It is considered the perfect getaway destination in Florida. It is popular due to their pristine white beaches and lush foliage. 🏖️🌴"
    )
    st.write(
        "Change the Cumulative CO2 Amount to see how having that much CO2 in the atmosphere in 2100 will affect sea level rise. The figure will have blue cover the affected areas 🌊. You can see even with the default amount of CO2 (how much there is in 2025) there is about 0.5 meters of sea level rise - this varies a little based on the emulator selected."
    )
    st.write("This vacation spot could possibly be submerged! 🤿")

    # if selected_emulator == "Pattern Scaling":
    #     plot_dem(sea_level_rise, "Pattern Scaling", "Tampa")
    # if selected_emulator == "Gaussian Process":
    #     plot_dem(sea_level_rise, "Gaussian Process", "Tampa")
    # if "CNN-LTSM" == selected_emulator:
    #     plot_dem(sea_level_rise, "CNN-LTSM", "Tampa")
    # if "Random Forest" == selected_emulator:
    #     plot_dem(sea_level_rise, "Random Forest", "Tampa")

    # if selected_emulator == "Pattern Scaling":
    #     plot_dem(sea_level_rise,"Pattern Scaling", "Miami")
    # if selected_emulator == "Gaussian Process":
    #     plot_dem(sea_level_rise,"Gaussian Process", "Miami")
    # if "CNN-LTSM" == selected_emulator:
    #     plot_dem(sea_level_rise, "CNN-LTSM", "Miami")
    # if "Random Forest" == selected_emulator:
    #     plot_dem(sea_level_rise, "Random Forest", "Miami")

    # if selected_emulator == "Pattern Scaling":
    #     plot_dem(sea_level_rise,"Pattern Scaling", "Hobe Sound")
    # if selected_emulator == "Gaussian Process":
    #     plot_dem(sea_level_rise,"Gaussian Process", "Hobe Sound")
    # if "CNN-LTSM" == selected_emulator:
    #     plot_dem(sea_level_rise, "CNN-LTSM", "Hobe Sound")
    # if "Random Forest" == selected_emulator:
    #     plot_dem(sea_level_rise, "Random Forest", "Hobe Sound")

    st.subheader("Fort Myers Beach Sea Level Rise in 2100")

    if selected_emulator == "Pattern Scaling":
        plot_dem(sea_level_rise,"Pattern Scaling", "Fort Myers Beach")
    if selected_emulator == "Gaussian Process":
        plot_dem(sea_level_rise,"Gaussian Process", "Fort Myers Beach")
    if "CNN-LTSM" == selected_emulator:
        plot_dem(sea_level_rise, "CNN-LTSM", "Fort Myers Beach")
    if "Random Forest" == selected_emulator:
        plot_dem(sea_level_rise, "Random Forest", "Fort Myers Beach")

    st.write("Fort Myers Beach is part of the Cape Coral-Fort Myers, Florida Metropolitan Statistical Area. There are many fun celebrations that take place here.")

    st.write("Fort Myers Beach is also a popular tourist destination known for it's white sand beaches, calm waters, and stunning sunsets.")

    # if selected_emulator == "Pattern Scaling":
    #     plot_dem(sea_level_rise,"Pattern Scaling", "Titusville")
    # if selected_emulator == "Gaussian Process":
    #     plot_dem(sea_level_rise,"Gaussian Process", "Titusville")
    # if "CNN-LTSM" == selected_emulator:
    #     plot_dem(sea_level_rise, "CNN-LTSM", "Titusville")
    # if "Random Forest" == selected_emulator:
    #     plot_dem(sea_level_rise, "Random Forest", "Titusville")

    # if selected_emulator == "Pattern Scaling":
    #     plot_dem(sea_level_rise,"Pattern Scaling", "Cape Canaveral")
    # if selected_emulator == "Gaussian Process":
    #     plot_dem(sea_level_rise,"Gaussian Process", "Cape Canaveral")
    # if "CNN-LTSM" == selected_emulator:
    #     plot_dem(sea_level_rise, "CNN-LTSM", "Cape Canaveral")
    # if "Random Forest" == selected_emulator:
    #     plot_dem(sea_level_rise, "Random Forest", "Cape Canaveral")

    st.subheader("Courtenay Sea Level Rise in 2100")

    if selected_emulator == "Pattern Scaling":
        plot_dem(sea_level_rise,"Pattern Scaling", "Courtenay")
    if selected_emulator == "Gaussian Process":
        plot_dem(sea_level_rise,"Gaussian Process", "Courtenay")
    if "CNN-LTSM" == selected_emulator:
        plot_dem(sea_level_rise, "CNN-LTSM", "Courtenay")
    if "Random Forest" == selected_emulator:
        plot_dem(sea_level_rise, "Random Forest", "Courtenay")

    st.write("Courtenay is within a few miles from the Kennedy Space Center and Merrit Island National Wildlife Refuge.")

    # if selected_emulator == "Pattern Scaling":
    #     plot_dem(sea_level_rise,"Pattern Scaling", "Cedar Key")
    # if selected_emulator == "Gaussian Process":
    #     plot_dem(sea_level_rise,"Gaussian Process", "Cedar Key")
    # if "CNN-LTSM" == selected_emulator:
    #     plot_dem(sea_level_rise, "CNN-LTSM", "Cedar Key")
    # if "Random Forest" == selected_emulator:
    #     plot_dem(sea_level_rise, "Random Forest", "Cedar Key")

if __name__ == "__main__":
    main()
