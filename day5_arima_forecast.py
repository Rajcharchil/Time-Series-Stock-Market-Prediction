import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Step 1: Load & preprocess
df = yf.download('RELIANCE.NS', start='2015-01-01', end='2024-12-31')
df = df[['Close']].dropna()
df.index = pd.to_datetime(df.index)
monthly_df = df['Close'].resample('ME').mean().dropna()

# Step 2: Apply Differencing
diff_data = monthly_df.diff().dropna()

# Step 3: Train ARIMA model (p=1, d=1, q=1 as default to start)
model = ARIMA(monthly_df, order=(1,1,1))
model_fit = model.fit()

# Step 4: Forecast next 12 months
forecast = model_fit.forecast(steps=12)
print("\n📊 Next 12 months forecast:")
print(forecast)

# Step 5: Plot original + forecast
plt.figure(figsize=(10, 4))
plt.plot(monthly_df, label='Actual', color='blue')
plt.plot(forecast.index, forecast, label='Forecast', color='red', linestyle='dashed')
plt.title("ARIMA Forecast for Reliance Stock")
plt.xlabel("Date")
plt.ylabel("Price (INR)")
plt.legend()
plt.grid(True)
plt.show()
