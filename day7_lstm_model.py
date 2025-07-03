import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# Step 1: Load data
df = yf.download('RELIANCE.NS', start='2015-01-01', end='2024-12-31')
df = df[['Close']].dropna()
df.index = pd.to_datetime(df.index)

# Step 2: Scale data to range 0-1
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

# Step 3: Create sequences
X = []
y = []

sequence_length = 60  # last 60 days → next day prediction

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i, 0])
    y.append(scaled_data[i, 0])

X = np.array(X)
y = np.array(y)

# Step 4: Reshape X for LSTM [samples, time_steps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Step 5: Split into training & testing
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Step 6: Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()

# Step 7: Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

# Step 8: Predict and inverse scale
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
y_test_scaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Step 9: Evaluate
rmse = np.sqrt(mean_squared_error(y_test_scaled, predictions))
print(f"\n✅ LSTM RMSE: {rmse:.2f}")

# Step 10: Plot the results
plt.figure(figsize=(12,5))
plt.plot(y_test_scaled, label='Actual Price')
plt.plot(predictions, label='Predicted Price', linestyle='dashed')
plt.title('LSTM Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
