import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# 1. Load & prepare data
df = pd.read_csv(
    "C:\\Users\\vikramp\\OneDrive - School Health Corporation\\Desktop\\PhD Qualifying Exam\\.venv\\PhD-Qualifying-Exam-Files-Vikram\\Q4_Code & Results\\5-Year_Product_Demand_Data.csv",
    parse_dates=["Month"]
)
df = df.rename(columns={"Month": "ds", "Demand": "y"})

# 2. Split into train (first 48 months) and test (last 12 months)
train = df.iloc[:48].copy()
test  = df.iloc[48:].copy()

# 3. Instantiate Prophet and add regressors
m = Prophet()
m.add_regressor("AdSpend")
m.add_regressor("CompetitorPrice")

# 4. Fit on training data
m.fit(train)

# 5. Build a future DataFrame for the test period,
#    including your two exogenous columns
future = test[["ds", "AdSpend", "CompetitorPrice"]].reset_index(drop=True)

# 6. Make forecast
forecast = m.predict(future)

# 7. Plot actual vs. forecast
plt.figure(figsize=(10,5))
plt.plot(train.ds, train.y,   label="Train")
plt.plot(test.ds,  test.y,    label="Actual (Test)", c="C2")
plt.plot(forecast.ds, forecast.yhat, label="Prophet Forecast", c="C3")
plt.fill_between(forecast.ds,
                 forecast.yhat_lower,
                 forecast.yhat_upper,
                 color="C3", alpha=0.2)
plt.legend()
plt.xlabel("Month")
plt.ylabel("Demand")
plt.title("Prophet + Regressors: Forecast vs Actual")
plt.show()

# 8. Evaluate performance
y_true = test.y.values
y_pred = forecast.yhat.values

rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mae  = mean_absolute_error(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

print(f"Prophet + regressors RMSE : {rmse:.2f}")
print(f"Prophet + regressors MAE  : {mae:.2f}")
print(f"Prophet + regressors MAPE : {mape:.2f}%")

# 9. (Optional) Inspect coefficients / regressor effects
# Prophet doesn’t give β’s directly, but you can examine the
# forecast components to see how each regressor contributes:
fig = m.plot_components(forecast)
plt.show()
