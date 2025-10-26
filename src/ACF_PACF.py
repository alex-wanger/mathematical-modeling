import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import acf

# Load CSV
nyc_df = pd.read_csv("nyc_gas_prices_2011.csv")
nyc_df['Date'] = pd.to_datetime(nyc_df['Date'])
nyc_df.set_index('Date', inplace=True)

# Second difference (d=2)
nyc_diff2 = nyc_df['NYC_Price'].diff().diff().dropna()

# Compute ACF values
acf_values = acf(nyc_diff2, nlags=50)  # adjust nlags if you want more
print("ACF values (lag 0 to 50):")
for i, val in enumerate(acf_values):
    print(f"Lag {i}: {val}")

# Plot ACF
plt.figure(figsize=(12,5))
plot_acf(nyc_diff2, lags=40, alpha=0.05)
plt.title('ACF of Second-Differenced Series (for MA term)')
plt.show()

# Plot PACF
plt.figure(figsize=(12,5))
plot_pacf(nyc_diff2, lags=20, method='ywm', alpha=0.05)
plt.title('PACF of Second-Differenced Series (for AR term)')
plt.show()
