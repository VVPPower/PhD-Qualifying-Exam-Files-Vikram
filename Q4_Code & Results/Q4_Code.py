"""
   Question 4 - Part I (35 pts): Univariate Time Series Modeling
    In this part, you will analyze the time series data of product demand over a 5-year period.
    You will perform the following tasks:
    1. Load the dataset and plot the time series to inspect trend, seasonality, and potential structural breaks.
    2. Decompose the series using STL (or classical decomposition) to identify trend, seasonality, and residuals.
    3. Plot the decomposed components to visualize the trend, seasonality, and residuals.
    4. Discuss the observed trend, seasonality, and any structural breaks in the data.
# Import necessary libraries
# Note: 1. Data Preparation and Exploratory Analysis

    """

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose



# Load the dataset (a CSV with 'Month' and 'Demand' columns)
df = pd.read_csv("5-Year_Product_Demand_Data.csv", parse_dates=["Month"], index_col="Month")
df = df.sort_index()  # Ensure the data is in chronological order

# Plot the time series to inspect trend, seasonality, and potential structural breaks
plt.figure(figsize=(10, 4))
plt.plot(df.index, df["Demand"], label="Demand")
plt.title("Product Demand Over 5 Years")
plt.xlabel("Month")
plt.ylabel("Demand")
plt.legend()
plt.show()

# Decompose the series using STL (or classical decomposition)
decomposition = seasonal_decompose(df["Demand"], model="additive", period=12)  # assuming monthly data
fig = decomposition.plot()
fig.set_size_inches(12, 8)
plt.show()
