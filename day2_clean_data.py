import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Download data
df = yf.download('RELIANCE.NS', start='2015-01-01', end='2024-12-31')

# Step 2: Check for missing values
print("🔍 Missing values per column:\n")
print(df.isnull().sum())

# Step 3: Remove rows where 'Close' is missing (main column)
df_cleaned = df[df['Close'].notnull()]

# Alternative: If you want to fill missing values with forward-fill
# df_cleaned = df.fillna(method='ffill')

# Step 4: Recheckpip install statsmodels

print("\n✅ Cleaned Data Missing Values:")
print(df_cleaned.isnull().sum())

# Step 5: Plot Cleaned Close Prices
plt.figure(figsize=(10, 4))
plt.plot(df_cleaned['Close'], color='green', label='Cleaned Close Price')
plt.title("Reliance Stock (Cleaned)")
plt.xlabel("Date")
plt.ylabel("Price (INR)")
plt.grid(True)
plt.legend()
plt.show()
