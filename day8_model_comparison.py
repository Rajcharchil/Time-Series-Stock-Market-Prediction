## Compare ARIMA vs LSTM (Graph + RMSE Table)

import matplotlib.pyplot as plt
import pandas as pd

# Replace with your actual values
rmse_arima = 148.76
rmse_lstm = 46.02

models = ['ARIMA', 'LSTM']
rmse_values = [rmse_arima, rmse_lstm]

plt.figure(figsize=(8,4))
plt.bar(models, rmse_values, color=['blue', 'green'])
plt.title('Model RMSE Comparison')
plt.ylabel('RMSE (Lower is Better)')
plt.grid(axis='y')
for i, v in enumerate(rmse_values):
    plt.text(i, v + 2, f"{v:.2f}", ha='center')
plt.tight_layout()
plt.show()

# Create a DataFrame for better visualization
comparison_df = pd.DataFrame({
    'Model': models,
    'RMSE': rmse_values
})
print(comparison_df)