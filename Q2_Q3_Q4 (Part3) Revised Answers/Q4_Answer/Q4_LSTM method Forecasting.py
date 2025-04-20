import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error


# 1. Load & prepare data
df = pd.read_csv(
     "C:\\Users\\vikramp\\OneDrive - School Health Corporation\\Desktop\\PhD Qualifying Exam\\.venv\\PhD-Qualifying-Exam-Files-Vikram\\Q4_Code & Results\\5-Year_Product_Demand_Data.csv",
    parse_dates=["Month"]
).sort_values("Month").reset_index(drop=True)

# 2. Build sequences: last 12 months of [Demand, AdSpend, CompetitorPrice] → predict next Demand
window = 12
features = df[["Demand", "AdSpend", "CompetitorPrice"]].values
X, y = [], []
for i in range(window, len(features)):
    X.append(features[i-window:i, :])  # shape (window, 3)
    y.append(features[i, 0])           # Demand at time i
X, y = np.array(X), np.array(y)

# 3. Train/test split
#   • first 48 months for training → (48 - 12) = 36 samples  
#   • last 12 months for test
n_train = 48 - window
X_train, y_train = X[:n_train], y[:n_train]
X_test,  y_test  = X[n_train:], y[n_train:]

# 4. Build & compile LSTM
model = Sequential([
    LSTM(50, input_shape=(window, 3), return_sequences=False),
    Dense(1)
])
model.compile(optimizer="adam", loss="mse")

# 5. Train (with 10% validation)
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=8,
    validation_split=0.1,
    verbose=1
)

# 6. Forecast
y_pred = model.predict(X_test).flatten()

# 7. Metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

print(f"LSTM + Exogenous RMSE : {rmse:.2f}")
print(f"LSTM + Exogenous MAE  : {mae:.2f}")
print(f"LSTM + Exogenous MAPE : {mape:.2f}%")

# 8. Plot Forecast vs. Actual
months = df["Month"].iloc[window + n_train : window + n_train + len(y_test)]
plt.figure(figsize=(10,5))
plt.plot(months, y_test,  marker="o", label="Actual")
plt.plot(months, y_pred,  marker="x", label="LSTM Forecast")
plt.title("LSTM with Exogenous Inputs")
plt.xlabel("Month")
plt.ylabel("Demand")
plt.legend()
plt.show()

# 9. Plot Training History
plt.figure(figsize=(8,4))
plt.plot(history.history["loss"],    label="Train Loss")
plt.plot(history.history["val_loss"],label="Val Loss")
plt.title("Training & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend()
plt.show()
