import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# Step 1: Load data
df = yf.download('RELIANCE.NS', start='2015-01-01', end='2024-12-31')
df = df[['Close']]
df.dropna(inplace=True)

# Step 2: Resample to monthly average
monthly_df = df['Close'].resample('M').mean()

# Step 3: ADF Test function
def adf_test(timeseries):
    print("📊 Augmented Dickey-Fuller Test:")
    result = adfuller(timeseries)
    labels = ['ADF Test Statistic', 'p-value', '# Lags Used', 'Number of Observations Used']
    for value, label in zip(result, labels):
        print(f"{label}: {value}")
    
    if result[1] <= 0.05:
        print("✅ Result: Data is stationary (Reject null hypothesis)")
    else:
        print("❌ Result: Data is NOT stationary (Fail to reject null hypothesis)")

# Step 4: Run ADF Test
adf_test(monthly_df)

# Step 5: Apply Differencing
diff_data = monthly_df.diff().dropna()

# Step 6: ADF Test Again
print("\n📉 After First Differencing:")
adf_test(diff_data)

# Step 7: Plot Differenced Series
plt.figure(figsize=(10, 4))
plt.plot(diff_data, color='orange', label='Differenced Close Price')
plt.title("Differenced Series (to achieve stationarity)")
plt.xlabel("Date")
plt.ylabel("Difference in Price")
plt.grid(True)
plt.legend()
plt.show()

