# Base logic generated with help from ChatGPT (OpenAI, 2025).
# The application used is visual studio code with copilot extension.
# Modified to fit assignment requirements and improved error handling.



"""
   Question 4 - Part I (35 pts): Univariate Time Series Modeling
    In this part, you will analyze the time series data of product demand over a 5-year period.
    You will perform the following tasks:
    1. Load the dataset and plot the time series to inspect trend, seasonality, and potential structural breaks.
    2. Decompose the series using STL (or classical decomposition) to identify trend, seasonality, and residuals.
    3. Plot the decomposed components to visualize the trend, seasonality, and residuals.
    4. Discuss the observed trend, seasonality, and any structural breaks in the data.

# Import necessary libraries
1. Data Preparation and Exploratory Analysis
# Explain how you detect any trend, seasonality, or potential structural breaks in the
series?
#  Which tests or approaches would you use to determine if differencing (nonseasonal
and/or seasonal) is required?

    """

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.graphics.tsaplots import plot_ccf




# Load the dataset (a CSV with 'Month' and 'Demand' columns)
df = pd.read_csv("C:\\Users\\vikramp\\OneDrive - School Health Corporation\\Desktop\\PhD Qualifying Exam\\.venv\\Q4_Code & Results\\5-Year_Product_Demand_Data.csv", parse_dates=["Month"], index_col="Month")

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
decomposition = seasonal_decompose(df["Demand"], model="additive", period=12)  # monthly data
fig = decomposition.plot()
fig.set_size_inches(12, 8)
plt.show()

"""
    In this part, you will analyze the time series data of product demand over a 5-year period.
   2. Model Identification and Specification
# Explain how you would choose the orders (p,d,q)(p, d, q) and seasonal orders
(P,D,Q)s(P, D, Q)_s.
# Discuss the roles of the ACF, PACF, and stationarity tests (e.g., Augmented Dickey-
Fuller) in guiding your model selection.
    """

from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Apply the Augmented Dickey-Fuller (ADF) test to assess stationarity (Differencing and Stationarity Check)
result = adfuller(df["Demand"])
print("ADF Statistic: %f" % result[0])
print("p-value: %f" % result[1])

# A p-value above 0.05 suggests the series is non-stationary and might require differencing.

# If differencing is applied, we can check the differenced series using ACF and PACF plots.
differenced_series = df["Demand"].diff().dropna()

# Plot ACF and PACF to inspect nonseasonal autocorrelations
fig, axes = plt.subplots(1, 2, figsize=(16, 4))
plot_acf(differenced_series, ax=axes[0], lags=29)  # Reduce lags to 29 (less than half of 60)
plot_pacf(differenced_series, ax=axes[1], lags=29)  # Reduce lags to 29 (less than half of 60)
plt.show()


# Seasonal differencing can be considered if significant autocorrelation is observed at seasonal lags.

"""
    In this part, you will analyze the time series data of product demand over a 5-year period.
   3. Estimation and Diagnostics
# Find the best parameters for the ARIMA/SARIMA model.
# Explain how you would assess whether residuals are white noise, including the use
of residual plots and statistical tests.
    """
# Define the SARIMA model with initial orders
model = sm.tsa.statespace.SARIMAX(df["Demand"],
                                  order=(1, 1, 1),
                                  seasonal_order=(0, 1, 1, 12),
                                  enforce_stationarity=False,
                                  enforce_invertibility=False)
results = model.fit(disp=False)
print(results.summary())

# Plot residuals
residuals = results.resid
plt.figure(figsize=(10, 4))
plt.plot(residuals)
plt.title("Residuals of the SARIMA Model")
plt.xlabel("Date")
plt.ylabel("Residuals")
plt.show()

# Plot ACF of residuals
plot_acf(residuals.dropna(), lags=50)
plt.show()

# Perform the Ljung-Box test on residuals
from statsmodels.stats.diagnostic import acorr_ljungbox
lb_test = acorr_ljungbox(residuals.dropna(), lags=[10], return_df=True)
print(lb_test)

"""
   4. Testing and Model Validation
# Generate forecasts for 5th year, including prediction intervals. Show your results in a
plot and give the values also in a table.
# Compare the results with the actuals or test data by plotting the predicted values
and the test data (actuals) with the time series.
# Describe which performance metrics (e.g., RMSE, MAE, MAPE) you would use to
evaluate the model, and why. Give the results of your analysis.
    """
# Define training and test sets (assuming the test set is the 5th year)
train = df[df.index.year < df.index.year.unique()[-1]]  # all years except the last
test = df[df.index.year == df.index.year.unique()[-1]]    # the 5th year

# Refit the model on the training data
model_train = sm.tsa.statespace.SARIMAX(train["Demand"],
                                        order=(1, 1, 1),
                                        seasonal_order=(0, 1, 1, 12),
                                        enforce_stationarity=False,
                                        enforce_invertibility=False)
results_train = model_train.fit(disp=False)

# Forecast for the test period (5th year)
forecast_obj = results_train.get_forecast(steps=len(test))
forecast = forecast_obj.predicted_mean
conf_int = forecast_obj.conf_int()

# Plotting the forecasts vs. actual test data
plt.figure(figsize=(12, 6))
plt.plot(train.index, train["Demand"], label="Training Data")
plt.plot(test.index, test["Demand"], label="Actual Demand (Test)", color="green")
plt.plot(forecast.index, forecast, label="Forecast", color="red")
plt.fill_between(forecast.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                 color='pink', alpha=0.3, label="95% Prediction Interval")
plt.xlabel("Date")
plt.ylabel("Demand")
plt.title("5th Year Forecast vs. Actual Demand")
plt.legend()
plt.show()

# Creating a table of forecasted values
forecast_df = pd.DataFrame({
    "Forecast": forecast,
    "Lower Bound": conf_int.iloc[:, 0],
    "Upper Bound": conf_int.iloc[:, 1]
})
print(forecast_df.head())

# Calculate performance metrics
rmse = np.sqrt(mean_squared_error(test["Demand"], forecast))
mae = mean_absolute_error(test["Demand"], forecast)
mape = np.mean(np.abs((test["Demand"] - forecast) / test["Demand"])) * 100

print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"MAPE: {mape:.2f}%")


"""
   Question 4 - Part II (35 pts): Incorporating Exogenous Variables
    In this part, you will extend your analysis by incorporating exogenous variables (X) into your
    SARIMA model. You will perform the following tasks:
    1. Load the dataset with exogenous variables and plot the time series to inspect trend, seasonality,
    and potential structural breaks.
    Now assume you wish to incorporate the external factors that potentially influence product
demand such as advertising spend and competitor’s price in your model using an ARIMAX or
SARIMAX framework.

1. Model Formulation with Exogenous Regressors
# How would you determine whether each exogenous variable (and its lags) should be
included in the model?
# Present a general form of the ARIMAX/SARIMAX equation, highlighting how external
variables enter the model.

    """


# Suppose 'Advertising' and 'CompetitorPrice' are columns in your DataFrame alongside 'Demand'
# Plot cross-correlation between demand and advertising spend
plot_ccf(df["Demand"].dropna(), df["AdSpend"].dropna(), lags=30)
plt.title("Cross-Correlation: Demand vs. Advertising Spend")
plt.show()

"""

2. Model Parameter Estimation, Diagonostic and stability
# Explain the challenges posed by multicollinearity among explanatory variables and
how you would diagnose or mitigate them.
# Find the best ARIMAX/SARIMAX model using the test data.
# Describe how you would verify that adding exogenous variables genuinely improves
the model (e.g., comparing information criteria, checking residual plots).

    """

# Suppose X is a DataFrame with the exogenous variables
X = df[["AdSpend", "CompetitorPrice"]]
X = sm.add_constant(X)

# Calculate VIF for each explanatory variable
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data)


# Define the SARIMAX model with exogenous variables (assuming 'Advertising' and 'CompetitorPrice' are available)
exog = df[["AdSpend", "CompetitorPrice"]]

model_exog = sm.tsa.statespace.SARIMAX(df["Demand"],
                                       order=(1, 1, 1),
                                       seasonal_order=(0, 1, 1, 12),
                                       exog=exog,
                                       enforce_stationarity=False,
                                       enforce_invertibility=False)
results_exog = model_exog.fit(disp=False)
print(results_exog.summary())

# After fitting the model, examine residuals and compare AIC/BIC
print("ARIMAX AIC:", results_exog.aic)
print("ARIMAX BIC:", results_exog.bic)

# Plot residuals of the ARIMAX model to ensure they are white noise
residuals_exog = results_exog.resid
plt.figure(figsize=(10, 4))
plt.plot(residuals_exog)
plt.title("Residuals of the ARIMAX Model")
plt.xlabel("Date")
plt.ylabel("Residuals")
plt.show()

# Plot ACF of residuals for the ARIMAX model
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(residuals_exog.dropna(), lags=50)
plt.show()

"""
3. Model Parameter Estimation, Diagnostic and stability
# Generate the forecast for 5th year and compare with the test data. Generate the
performance metrics and plots for this model.
# Compare with the univariate model and determine which model is better. Justify
your answer
    """
# Split data into training and test sets (assuming test set is the 5th year)
train = df[df.index.year < df.index.year.unique()[-1]]
test = df[df.index.year == df.index.year.unique()[-1]]

# Assume exogenous variables are available in the dataset
exog_train = train[["AdSpend", "CompetitorPrice"]]
exog_test = test[["AdSpend", "CompetitorPrice"]]

# Fit the SARIMAX model with exogenous variables on the training data
model_exog = sm.tsa.statespace.SARIMAX(train["Demand"],
                                       order=(1, 1, 1),
                                       seasonal_order=(0, 1, 1, 12),
                                       exog=exog_train,
                                       enforce_stationarity=False,
                                       enforce_invertibility=False)
results_exog = model_exog.fit(disp=False)
print(results_exog.summary())

# Generate forecasts for the test period, providing the exogenous data for forecasting
forecast_obj_exog = results_exog.get_forecast(steps=len(test), exog=exog_test)
forecast_exog = forecast_obj_exog.predicted_mean
conf_int_exog = forecast_obj_exog.conf_int()



# Plot forecasts against actual test data
plt.figure(figsize=(12, 6))
plt.plot(train.index, train["Demand"], label="Training Data")
plt.plot(test.index, test["Demand"], label="Actual Demand (Test)", color="green")
plt.plot(forecast_exog.index, forecast_exog, label="Forecast (ARIMAX)", color="red")
plt.fill_between(forecast_exog.index, conf_int_exog.iloc[:, 0], conf_int_exog.iloc[:, 1],
                 color='pink', alpha=0.3, label="95% Prediction Interval")
plt.xlabel("Date")
plt.ylabel("Demand")
plt.title("5th Year Forecast vs. Actual Demand (ARIMAX)")
plt.legend()
plt.show()

# Generate a table of forecast values
forecast_exog_df = pd.DataFrame({
    "Forecast": forecast_exog,
    "Lower Bound": conf_int_exog.iloc[:, 0],
    "Upper Bound": conf_int_exog.iloc[:, 1]
})
print(forecast_exog_df.head())

# Calculate performance metrics for the ARIMAX model
rmse_exog = np.sqrt(mean_squared_error(test["Demand"], forecast_exog))
mae_exog = mean_absolute_error(test["Demand"], forecast_exog)
mape_exog = np.mean(np.abs((test["Demand"] - forecast_exog) / test["Demand"])) * 100

print(f"ARIMAX RMSE: {rmse_exog:.2f}")
print(f"ARIMAX MAE: {mae_exog:.2f}")
print(f"ARIMAX MAPE: {mape_exog:.2f}%")

# (Assume similar metrics are calculated for the univariate SARIMA model)

"""
4. Forecasting and Interpretation
# If you have (or can predict) the future values of these exogenous variables, describe
how you would generate and interpret out-of-sample forecasts.
# Offer guidelines on presenting model results to both technical and non-technical
audiences, focusing on how changes in each exogenous variable affect demand.
    """
# we have future values for the exogenous variables (Advertising and CompetitorPrice)
# for the next 12 months (or the 5th year period)

advertising_values = [
    20.99, 22.22, 25.63, 28.05, 23.86, 22.03, 23.16, 19.03, 14.73, 16.09,
    14.74, 16.57, 20.48, 18.67, 20.88, 23.88, 22.30, 23.13, 18.18, 14.68,
    18.60, 14.55, 15.80, 14.65, 18.91, 22.72, 22.03, 25.75, 23.13, 21.92,
    18.80, 21.20, 15.64, 12.88, 17.31, 15.06, 20.42, 18.58, 21.67, 25.39,
    25.81, 22.84, 19.77, 16.90, 12.71, 13.56, 14.75, 19.61, 20.69, 18.97,
    24.98, 24.23, 22.98, 23.72, 22.06, 19.36, 13.99, 14.38, 16.33, 19.45
]

competitor_price_values = [
    49.52, 49.91, 49.09, 49.10, 51.21, 51.86, 50.53, 51.70, 51.16, 50.25,
    51.36, 52.64, 51.16, 52.86, 48.78, 52.32, 51.69, 51.40, 51.89, 49.91,
    51.78, 52.46, 53.68, 51.78, 51.59, 52.00, 53.52, 53.03, 52.27, 53.41,
    53.10, 54.07, 52.50, 52.97, 53.01, 52.04, 53.90, 53.96, 53.81, 53.67,
    52.58, 53.68, 53.86, 53.50, 54.24, 54.90, 56.49, 54.87, 55.06, 54.83,
    53.08, 55.07, 55.26, 57.76, 55.21, 55.80, 55.57, 54.53, 56.94, 56.65
]

# Create a date range for 60 periods with monthly frequency, starting from a chosen date.
future_index = pd.date_range(start="2020-01-01", periods=60, freq="MS")

# Create a DataFrame for future exogenous variables using the provided values.
future_exog = pd.DataFrame({
    "AdSpend": advertising_values,
    "CompetitorPrice": competitor_price_values
}, index=future_index)

print(future_exog.head())


# Select only the last 12 months (5th year) from the full 60-period exogenous DataFrame.
future_exog_test = future_exog.iloc[-12:]

# Generate the forecast for the 5th year using these 12 rows of exogenous data.
forecast_obj_future = results_exog.get_forecast(steps=12, exog=future_exog_test)
forecast_future = forecast_obj_future.predicted_mean
conf_int_future = forecast_obj_future.conf_int()

print(forecast_future)



"""
Question 4 - Part III (30 pts): Research other forecasting models
    In this part, I will explore alternative forecasting models and techniques. 

Describe an alternative technique for forecasting product demand and apply it to your dataset.
Compare and contrast its results with your previous findings. Which model performs best for your
dataset, and why?

    """

from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Fit Holt-Winters model on the training data
hw_model = ExponentialSmoothing(train["Demand"], trend="add", seasonal="add", seasonal_periods=12)
hw_results = hw_model.fit(optimized=True)

# Generate forecasts for the test period (5th year)
hw_forecast = hw_results.forecast(steps=len(test))

# Plot the Holt-Winters forecast against actual test data
plt.figure(figsize=(12,6))
plt.plot(train.index, train["Demand"], label="Training Data")
plt.plot(test.index, test["Demand"], label="Actual Demand (Test)", color="green")
plt.plot(test.index, hw_forecast, label="Holt-Winters Forecast", color="orange")
plt.xlabel("Date")
plt.ylabel("Demand")
plt.title("Holt-Winters Forecast vs. Actual Demand")
plt.legend()
plt.show()

# Calculate performance metrics for the Holt-Winters model
rmse_hw = np.sqrt(mean_squared_error(test["Demand"], hw_forecast))
mae_hw = mean_absolute_error(test["Demand"], hw_forecast)
mape_hw = np.mean(np.abs((test["Demand"] - hw_forecast) / test["Demand"])) * 100

print(f"Holt-Winters RMSE: {rmse_hw:.2f}")
print(f"Holt-Winters MAE: {mae_hw:.2f}")
print(f"Holt-Winters MAPE: {mape_hw:.2f}%")
