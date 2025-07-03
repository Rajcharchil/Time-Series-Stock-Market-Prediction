import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error
import numpy as np

# Step 1: Load and resample data
df = yf.download('RELIANCE.NS', start='2015-01-01', end='2024-12-31')
df = df[['Close']].dropna()
df.index = pd.to_datetime(df.index)
monthly_df = df['Close'].resample('ME').mean().dropna()

# Step 2: Split into train & test
train = monthly_df[:-12]
test = monthly_df[-12:]

# Step 3: Fit Auto ARIMA
model = auto_arima(train, seasonal=False, stepwise=True, trace=True)
print("\nBest Model Order (p,d,q):", model.order)

# Step 4: Predict
forecast = model.predict(n_periods=12)
forecast = pd.Series(forecast, index=test.index)

# Step 5: Plot predictions vs actual
plt.figure(figsize=(10,4))
plt.plot(train, label='Train Data', color='blue')
plt.plot(test, label='Actual (Test)', color='green')
plt.plot(forecast, label='Forecast (Auto ARIMA)', color='red', linestyle='--')
plt.title('Auto ARIMA Forecast vs Actual')
plt.xlabel('Date')
plt.ylabel('Price (INR)')
plt.legend()
plt.grid(True)
plt.show()

# Step 6: Evaluation
mape = mean_absolute_percentage_error(test, forecast)
rmse = np.sqrt(mean_squared_error(test, forecast))

print(f"\n📊 Evaluation Metrics:")
print(f"✅ MAPE: {mape*100:.2f}%")
print(f"✅ RMSE: {rmse:.2f}")

# Step 7: Create comparison DataFrame

comparison = pd.DataFrame({
    'Actual': test,
    'Forecast': forecast
}).reset_index()
