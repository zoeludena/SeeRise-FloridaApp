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
    co2 = st.sidebar.slider(
        "Cumulative CO2 Amount (GtCO2) in 2100", 0, max_co2, 4520, 10
    )
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


def map_ui():
    st.sidebar.markdown("# Select Location 🗺️")
    locations = ["Sanibel Island", "Miami", "Fort Myers Beach", "Audubon", "Your Choice"]
    selected_location = st.sidebar.selectbox("Choose a location:", locations, index=0)
    return selected_location


def directions():
    st.write(
        "Change the Cumulative CO2 Amount in 2100 to see how having that much CO2 in the atmosphere in 2100 will affect sea level rise. The affected areas in the figure will be covered by blue 🌊."
    )
    st.write(
        "You can see that, even with the default amount of CO2 (4520 Gigatons), there is about 0.5 meters of sea level rise - this varies a little based on the emulator selected."
    )


def main():
    st.title("🌊 SeeRise: Visualizing Emulated Sea Level Rise on Florida")

    co2 = emissions_ui()

    selected_emulator, emulator_colors = emulator_ui()

    location = map_ui()

    st.subheader(f"Selected emulator: {selected_emulator}")

    st.write("👋 Hello there! Welcome to our application that predicts sea level rise!")

    st.write(
        "This application predicts the sea level rise under the assumption of SSP 245's values in 2100. The only variable you are controlling is cumulative carbon dioxide. Your starting point is the amount of carbon dioxide (in gigatons) predicted for 2100 by SSP 245 (4520 gigatons). There is 3340 gigatons of carbon dioxide in 2025."
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
            "RCP4.5 or 4.5 W/m² Radiative Forcing by 2100 means a stabilization scenario where emissions peak around mid-21-century and then decline. It assumes that mitigation policies slow global warming but do not fully stop it. This will lead to a moderate level of warming (~2.5-3°C above pre-industrial levels by 2100)."
        )
        st.write(
            "This future allows the temperature to rise a little more that what the Paris Agreement hopes for."
        )

        # st.markdown('You can select different emulators based off of <a href="https://ftp.labins.org/dem/fl/" target="_blank">ClimateBench</a> 🌎. hese emulators use your input of cumulative carbon dioxide and the SSP 245\'s values for other greenhouse gases in 2025 to predict the temperature 🌡️. From there, we will predict the sea level rise using linear regression.', unsafe_allow_html=True)
    st.write(
        "You can select different emulators based off of ClimateBench 🌎. These emulators use your input of cumulative carbon dioxide and the SSP 245's values for other greenhouse gases in 2025 to predict the temperature 🌡️. From there, we will predict the sea level rise using linear regression."
    )

    with st.expander("🤖 Learn about your emulator:"):
        if selected_emulator == "Pattern Scaling":
            st.write(
                'The pattern scaling model is the simplest emulator at our disposal. The model consists of many linear regression models trained on global mean temperature in different emission scenarios. These models regress desired variables (precipitation, diurnal temperature range, etc.) on global mean temperature which is the "scaling" element of the model. Once trained, the model takes a vector of global mean temperatures from a particular emission scenario, and predicts the desired variables using the inputs. This model is powerful yet simple because it can predict local values of particular variables using only globally averaged inputs.'
            )
        elif selected_emulator == "Gaussian Process":
            st.write(
                "A Gaussian Process (GP) model is a probabilistic framework ideal for regression and classification tasks. GPs model functions by defining a prior characterized by a mean function, m(x), representing the expected value at x, and a covariance function, k(x, x'), which measures similarity between inputs x and x'. Using Bayesian inference, GPs update this prior with training data to produce a posterior distribution. For new inputs, predictions are made as a distribution with a mean (most likely value) and variance (uncertainty estimate)."
            )
            st.write(
                "GP models are well-suited for climate prediction. Climate systems are governed by complex, smooth, and often nonlinear relationships, which GPs can model through appropriately chosen kernels. Moreover, their ability to provide uncertainty estimates is invaluable when working with limited or noisy climate data, as these estimates can highlight regions where the model is less confident in its predictions. Finally, the interpretability of GP models aligns well with scientific practices, allowing researchers to explore the relationships captured by the covariance function and gain insights into the modeled climate dynamics."
            )
        elif selected_emulator == "Random Forest":
            st.write(
                "Random Forest is an ensemble method that aggregates the predictions of multiple decision trees to enhance predictive performance. Decision trees, as the base models, are particularly effective at capturing non-linear relationships and interactions between variables but are prone to overfitting. Random Forest addresses this limitation by averaging the predictions of all individual trees, which reduces variance and increases robustness. This makes it well-suited for climate model emulation, where separate models are often developed for multiple target variables."
            )
            st.write(
                "One key advantage of Random Forest in climate model emulation is its interpretability, which aids in informing decision-making. While a common drawback of Random Forest is its inability to extrapolate beyond the range of training data, this is not a significant concern in this context. Relevant predictions in climate modeling typically lie within the range defined by historical climate data and plausible scenarios, such as the low-emissions SSP 126 and high-emissions SSP 585 pathways. This makes Random Forest an effective and practical choice for emulating climate models."
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
        "For your convenience, we have determined that the length of an iPhone 📱 is around 146.6 mm to help you with having a better understanding of the extent of sea level rise."
    )

    st.write(
        "The figure above shows you a box plot of predicted sea level rise with uncertainty. The median (50th percentile) is a reasonable estimate of sea level rise."
    )

    st.write(
        "😱 \"Wow, that's scary!\" However, what's even more concerning might be the following observation: land slopes. This means the sea level rise will cause ocean water to flow inland, shrinking our coastlines."
    )

    st.subheader("Visualizing Sea Level Rise on Florida's Coast in 2100")

    st.write(
        "Below, you will see our visualizations for select locations on Florida's coastline. We used a DEM file to provide us with the elevation data of these locations."
    )

    with st.expander("What is DEM? 🤔"):
        st.write(
            "DEM stands for Digital Elevation Model. It aims to recreate the topographic surface of the Earth excluding buildings and foliage."
        )
        st.write(
            "We want to show the digital representation of the contours of earths surface to make it easier for you as a audience to visualize sea level rise. It is important to note that, in high residential areas, the DEM is not always accurate. For example, it has trouble with bridges that connect land masses."
        )
        st.write(
            "We obtained these DEM files from the Land Boundary Information System (LABINS), under mapping data."
        )

    st.write(
        "Your options for pre-set locations are: Sanibel Island, Miami, Fort Myers Beach, and Audubon."
    )
    
    st.write(
        "Some of these locations have greater elevations than 5 meters. However, to highlight the impact, we only color the 0 to 5 meter range. Anything above or below that range is going to appear white in the visualization."
    )

    st.write("You can upload your own DEM file to visualize other areas!")

    if location == "Sanibel Island":

        st.subheader("Sanibel Island Sea Level Rise")

        directions()

        with st.expander("Click to see a labeled map:"):
            st.image("data/google_pics/sanibel_google.png")
            st.write(
                "This is a screenshot of Google Maps to help you better orient yourself."
            )

        if selected_emulator == "Pattern Scaling":
            plot_dem(
                sea_level_rise, "Pattern Scaling", "Sanibel Island", (0, 950, 0, 1300)
            )
        if selected_emulator == "Gaussian Process":
            plot_dem(
                sea_level_rise, "Gaussian Process", "Sanibel Island", (0, 950, 0, 1300)
            )
        if "CNN-LTSM" == selected_emulator:
            plot_dem(sea_level_rise, "CNN-LTSM", "Sanibel Island", (0, 950, 0, 1300))
        if "Random Forest" == selected_emulator:
            plot_dem(
                sea_level_rise, "Random Forest", "Sanibel Island", (0, 950, 0, 1300)
            )

        st.write(f"As a reminder, you are using the {selected_emulator} emulator.")

        # st.write("Above you can see Sanibel Island. It is considered the perfect getaway destination in Florida. It is popular due to their pristine white beaches and lush foliage. 🏖️🌴"
        # )
        # st.write("This vacation spot could possibly be submerged! 🤿")

        st.write(
            "We can see that Sanibel Island is almost halfway submerged when there is 4520 Gigatons of cumulative carbon dioxide in 2100. The part that is submerged happens to be J.N. Ding Darling National Wildlife Refuge and the Dunes."
        )

        st.write(
            "J.N. Ding Darling National Wildlife Refuge is one of the most important wildlife refuges in the US. It protects mangroves, bird species, and aquatic life. It is a critical habitat for the West Indian manatee and the smalltooth sawfish. It is a 7,600 acre protected area on Sanibel Island. It has been recognized as an Outstanding Florida Water - a title given to ecologically important regions that require extra protection."
        )

        st.write(
            "The Dunes is a golf and tennis club that allows people to play within a wildlife preserve."
        )

        st.write(
            "As cumulative carbon dioxide increases, the Clinic for the Rehabilitation of Wildlife and Sanibel Historical Museum and Village become endangered."
        )

    # if location == "Tampa":

    #     st.subheader("Tampa Sea Level Rise")

    #     directions()

    #     with st.expander("Click to see a labeled map:"):
    #         st.image("data/google_pics/tampa_google.png")
    #         st.write(
    #             "This is a screenshot of Google Maps to help you better orient yourself."
    #         )

    #     if selected_emulator == "Pattern Scaling":
    #         plot_dem(sea_level_rise, "Pattern Scaling", "Tampa", (150, 500, 100, 350))
    #     if selected_emulator == "Gaussian Process":
    #         plot_dem(sea_level_rise, "Gaussian Process", "Tampa", (150, 500, 100, 350))
    #     if "CNN-LTSM" == selected_emulator:
    #         plot_dem(sea_level_rise, "CNN-LTSM", "Tampa", (150, 500, 100, 350))
    #     if "Random Forest" == selected_emulator:
    #         plot_dem(sea_level_rise, "Random Forest", "Tampa", (150, 500, 100, 350))

    #     st.write(f"As a reminder, you are using the {selected_emulator} emulator.")

    #     st.write("Tampa is a major economic and cultural hub on Florida's west coast.")
    #     st.write(
    #         "Davis Islands, Tampa, was developed by D.P. Davis in the 1920s and is considered one of the most exclusive and desirable neighborhoods in the city. Although the coastline of Davis Islands is at a relatively lower elevation, we can see that it is not predicted to be severely affected by sea level rise, with only minor encroachment expected."
    #     )
    #     st.write(
    #         "The Port of Tampa Bay, to the east of Davis Islands, is the largest port in Florida and a key economic engine of the region. It is also not expected to be significantly impacted by sea level rise."
    #     )

    if location == "Miami":

        st.subheader("Miami Sea Level Rise")

        directions()

        with st.expander("Click to see a labeled map:"):
            st.image("data/google_pics/miami_google.png")
            st.write(
                "This is a screenshot of Google Maps to help you better orient yourself."
            )

        if selected_emulator == "Pattern Scaling":
            plot_dem(sea_level_rise, "Pattern Scaling", "Miami", (0, 1300, 400, 1300))
        if selected_emulator == "Gaussian Process":
            plot_dem(
                sea_level_rise, "Gaussian Process", "Miami", (400, 1300, 400, 1300)
            )
        if "CNN-LTSM" == selected_emulator:
            plot_dem(sea_level_rise, "CNN-LTSM", "Miami", (400, 1300, 400, 1300))
        if "Random Forest" == selected_emulator:
            plot_dem(sea_level_rise, "Random Forest", "Miami", (400, 1300, 400, 1300))

        st.write(f"As a reminder, you are using the {selected_emulator} emulator.")

        st.write(
            "Miami is a major cultural, financial, and tourism hub. It has 6,372,000 residents as of 2025."
        )
        st.write(
            "We can see that North Bay Village, Dodge Island, and the other islands around Miami are in danger."
        )
        st.write(
            "North Bay Village and the other islands (Star, Palm, and Hibiscus Islands) are known for their waterfront homes. It is home to celebrities, millionaires, and athletes. 🏠"
        )

        st.write(
            "People losing their homes seems to be a common trend among our visualizations. To make matters worse, the US's trade could be negatively impacted."
        )
        st.write(
            "Dodge Island is a major hub for global shipping, cargo, and tourism 📦. It plays a vital role in the global supply chain, supporting US imports and exports. PortMiami, located on Dodge Island, processes over 1 million units of cargo (twenty feet each) per year and generates about $43 billion in economic activity annually. This makes it one of the busiest ports in the Southeastern US. It provides faster access to Latin America and Caribbean markets than other East Coast ports. It is also connected to Florida East Coast Railway, allowing for rapid distribution. Over 240,000 jobs in Florida depend on PortMiami."
        )
        st.write("")

    if location == "Fort Myers Beach":

        st.subheader("Fort Myers Beach Sea Level Rise")

        directions()

        with st.expander("Click to see a labeled map:"):
            st.image("data/google_pics/fort_myers_google.png")
            st.write(
                "This is a screenshot of Google Maps to help you better orient yourself."
            )

        if selected_emulator == "Pattern Scaling":
            plot_dem(sea_level_rise, "Pattern Scaling", "Fort Myers Beach")
        if selected_emulator == "Gaussian Process":
            plot_dem(sea_level_rise, "Gaussian Process", "Fort Myers Beach")
        if "CNN-LTSM" == selected_emulator:
            plot_dem(sea_level_rise, "CNN-LTSM", "Fort Myers Beach")
        if "Random Forest" == selected_emulator:
            plot_dem(sea_level_rise, "Random Forest", "Fort Myers Beach")

        st.write(f"As a reminder, you are using the {selected_emulator} emulator.")

        st.write(
            "Fort Myers Beach is one of Florida's top beach destinations and holds famous festivals and large competitions. It is also of ecological and historical importance. It is home to Lovers Key State Park, which is a protected area with mangroves, manatees, and wildlife. The Mound House preserves the history of the Calusa Native Americans who lived there."
        )

        st.write(
            "We can see that San Carlos Island and the northern parts of Estero Island are at risk of submersion. This area is extremely vulnerable!"
        )

        st.write(
            "San Carlos Island serves as a hub for Florida's shrimping industry. Many charter fishing boats, marinas, and seafood packing houses operate there. It is popular for wildlife tours, dolphin-watching cruises, and eco-kayaking."
        )

        st.write(
            "Estero Island is a popular tourist destination. You can explore hiking trails, boardwalks, and birdwatching. There are 7 miles of soft, warm, white sand. It is popular for swimming, shelling, jet skiing, and parasailing."
        )

    if location == "Audubon":

        st.subheader("Audubon Sea Level Rise")

        directions()

        with st.expander("Click to see a labeled map:"):
            st.image("data/google_pics/audubon_google.png")
            st.write(
                "This is a screenshot of Google Maps to help you better orient yourself."
            )

        if selected_emulator == "Pattern Scaling":
            plot_dem(sea_level_rise, "Pattern Scaling", "Audubon", (200, 500, 0, 400))
        if selected_emulator == "Gaussian Process":
            plot_dem(sea_level_rise, "Gaussian Process", "Audubon", (200, 500, 0, 400))
        if "CNN-LTSM" == selected_emulator:
            plot_dem(sea_level_rise, "CNN-LTSM", "Audubon", (200, 500, 0, 400))
        if "Random Forest" == selected_emulator:
            plot_dem(sea_level_rise, "Random Forest", "Audubon", (200, 500, 0, 400))

        st.write(f"As a reminder, you are using the {selected_emulator} emulator.")

        st.write(
            "The Martin Andersen Beachline Expressway crosses through this region, carrying approximately 213 thousand vehicles a day and serving as a crucial connection for residents and visitors traveling from Florida's east coast beaches to Cape Canaveral. It also connects to routes to the John F. Kennedy Space Center."
        )

        st.write(
            "We can see that the Savannahs and Diana Shores, both residential areas, are at high risk. Severe flooding is likely following sea level rise, which could displace residents and damage homes."
        )

    # if selected_emulator == "Pattern Scaling":
    #     plot_dem(sea_level_rise,"Pattern Scaling", "Cedar Key")
    # if selected_emulator == "Gaussian Process":
    #     plot_dem(sea_level_rise,"Gaussian Process", "Cedar Key")
    # if "CNN-LTSM" == selected_emulator:
    #     plot_dem(sea_level_rise, "CNN-LTSM", "Cedar Key")
    # if "Random Forest" == selected_emulator:
    #     plot_dem(sea_level_rise, "Random Forest", "Cedar Key")

    if location == "Your Choice":
        st.subheader("Upload a DEM file to see sea level rise in an area of your choice!")

        st.markdown('You can create a DEM file of Florida from <a href="https://ftp.labins.org/dem/fl/" target="_blank">here</a>. If you do this please follow the directions below.', unsafe_allow_html=True)

        with st.expander("Click for Directions:"):
            st.write("You will notice there are many files here. Choose a location that intrigues you. For these directions I will choose Ozello.")
            st.markdown("1. You will want to download the file that ends in `.dem.sdts.tar.gz`.")
            st.write("2. You will then need to unzip the file.")
            st.markdown("3. You will go into the unzipped folder and then unzip the `.dem.sdts` file.")
            st.markdown("Once you are inside of the open `.dem.sdts` file you will notice a bunch of `.DDF` files. We now need to convert these `.DDF` files unto a DEM file.")
            st.markdown('4. You will need to download `sdts2dem.exe` from <a href="https://www2.cs.arizona.edu/projects/topovista/sdts2dem/" target="_blank">here</a>.', unsafe_allow_html=True)
            st.markdown("5. Move `sdts2dem.exe` to where your `.DDF` files are located.")
            st.write("6. Open a terminal/command prompt.")
            st.markdown("7. Navigate to the directory with your `.DDF` files and `sdts2dem.exe` file.")
            st.markdown("For example in your terminal write `cd Downloads\\1658320.dem.sdts`. You will need to change `1658320.dem.sdts` with your folder's name.")
            st.markdown("Look at the first four numbers of the `.DDF` files. We will refer to the numbers as `####`.")
            st.markdown("8. Type: `sdts2dem #### location_name`.")
            st.markdown("For example: `sdts2dem 8735 Ozello`.")
            st.markdown("Now you have a DEM file i.e. `Ozello.dem`. You can now upload the file!")

        st.markdown("Here is another <a href='https://gisgeography.com/top-6-free-lidar-data-sources/' target='_blank'>link</a> to GISGeography that provides the \"Top 6 Free LiDAR Data Sources\" to find DEM files.", unsafe_allow_html=True)

        file = st.file_uploader("Upload a DEM file (.dem)", type=["dem"])
        if selected_emulator == "Pattern Scaling":
            plot_dem(sea_level_rise,"Pattern Scaling", "Your Choice", uploaded_file=file)
        if selected_emulator == "Gaussian Process":
            plot_dem(sea_level_rise,"Gaussian Process", "Your Choice", uploaded_file=file)
        if "CNN-LTSM" == selected_emulator:
            plot_dem(sea_level_rise, "CNN-LTSM", "Your Choice", uploaded_file=file)
        if "Random Forest" == selected_emulator:
            plot_dem(sea_level_rise, "Random Forest", "Your Choice", uploaded_file=file)

if __name__ == "__main__":
    main()
