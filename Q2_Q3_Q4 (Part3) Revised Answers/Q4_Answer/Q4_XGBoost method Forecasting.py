# Supervised Forecasting with XGBoost and Lagged + Exogenous Features
# Dataset: 5‑Year Product Demand with AdSpend & CompetitorPrice :contentReference[oaicite:0]{index=0}&#8203;:contentReference[oaicite:1]{index=1}
# Method: Gradient Boosting Trees (XGBoost) per Chen & Guestrin (2016) :contentReference[oaicite:2]{index=2}&#8203;:contentReference[oaicite:3]{index=3}

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# 1. Load and sort data
df = pd.read_csv(
    "C:\\Users\\vikramp\\OneDrive - School Health Corporation\\Desktop\\PhD Qualifying Exam\\.venv\\PhD-Qualifying-Exam-Files-Vikram\\Q4_Code & Results\\5-Year_Product_Demand_Data.csv",
    parse_dates=["Month"]
)
df = df.sort_values("Month").reset_index(drop=True)

# 2. Create lag features for demand (1–12 months) and for exogenous vars (lags 1–3)
max_demand_lag = 12
exog_lags = [1, 2, 3]
for lag in range(1, max_demand_lag + 1):
    df[f"demand_lag_{lag}"] = df["Demand"].shift(lag)
for lag in exog_lags:
    df[f"AdSpend_lag_{lag}"]       = df["AdSpend"].shift(lag)
    df[f"CompetitorPrice_lag_{lag}"] = df["CompetitorPrice"].shift(lag)

# 3. Drop initial rows with NaNs from lagging
df_feat = df.dropna().reset_index(drop=True)

# 4. Split into train (first 36 after lag drop) & test (last 12) — aligns with months 1–48 vs 49–60
n_test  = 12
n_total = len(df_feat)          # should be 60 – 12 = 48
n_train = n_total - n_test      # = 36

train = df_feat.iloc[:n_train].copy()
test  = df_feat.iloc[n_train:].copy()

# 5. Define feature columns and target
feature_cols = (
    [f"demand_lag_{i}" for i in range(1, max_demand_lag + 1)] +
    [f"AdSpend_lag_{i}" for i in exog_lags] +
    [f"CompetitorPrice_lag_{i}" for i in exog_lags] +
    ["AdSpend", "CompetitorPrice"]
)
X_train, y_train = train[feature_cols], train["Demand"]
X_test,  y_test  = test[feature_cols],  test["Demand"]

# 6. Train XGBoost regressor
model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)

# 7. Predict on test set
y_pred = model.predict(X_test)

# 8. Evaluate performance
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f"XGBoost + lags RMSE : {rmse:.2f}")
print(f"XGBoost + lags MAE  : {mae:.2f}")
print(f"XGBoost + lags MAPE : {mape:.2f}%")

# 9. Plot actual vs. forecast
plt.figure(figsize=(10, 5))
plt.plot(train["Month"], train["Demand"], label="Train", alpha=0.6)
plt.plot(test["Month"],  y_test,             label="Actual (Test)", marker="o")
plt.plot(test["Month"],  y_pred,             label="XGBoost Forecast", marker="x")
plt.xlabel("Month")
plt.ylabel("Demand")
plt.title("XGBoost with Lagged & Exogenous Features")
plt.legend()
plt.show()
