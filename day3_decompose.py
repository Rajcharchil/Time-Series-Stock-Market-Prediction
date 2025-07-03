import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# Step 1: Load data
df = yf.download('RELIANCE.NS', start='2015-01-01', end='2024-12-31')
df = df[['Close']]
df.dropna(inplace=True)

# Step 2: Convert index to datetime (if not already)
df.index = pd.to_datetime(df.index)

# Step 3: Resample monthly (to make seasonality visible)
monthly_df = df['Close'].resample('M').mean()

# Step 4: Decompose the time series
result = seasonal_decompose(monthly_df, model='additive', period=12)

# Step 5: Plot components
result.plot()
plt.suptitle("Decomposition of Monthly Average Stock Price", fontsize=14)
plt.tight_layout()
plt.show()
