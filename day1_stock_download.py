import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Reliance ka stock data download karo
data = yf.download('RELIANCE.NS', start='2015-01-01', end='2024-12-31')

# First 5 rows print karo
print(data.head())

# Line chart dikhana - Closing price
plt.figure(figsize=(10,4))
plt.plot(data['Close'], label='Close Price')
plt.title("Reliance Stock Price")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()

