import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

nyc_df = pd.read_csv("nyc_gas_prices_2011.csv")
nyc_df['Date'] = pd.to_datetime(nyc_df['Date'])
nyc_df.set_index('Date', inplace=True)

nyc_diff1 = nyc_df['NYC_Price'].diff().dropna()

nyc_diff2 = nyc_diff1.diff().dropna()

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 15), sharex=True)

axes[0].plot(nyc_df['NYC_Price'], marker='o', color='blue')
axes[0].set_title('Original Series')
axes[0].set_ylabel('Price ($/gallon)')
axes[0].grid(True)

axes[1].plot(nyc_diff1, marker='o', color='red')
axes[1].set_title('First Difference (d=1)')
axes[1].set_ylabel('Price Change ($/gallon)')
axes[1].grid(True)

axes[2].plot(nyc_diff2, marker='o', color='green')
axes[2].set_title('Second Difference (d=2)')
axes[2].set_xlabel('Date')
axes[2].set_ylabel('Change of Change ($/gallon)')
axes[2].grid(True)

result = adfuller(nyc_diff2)

print('ADF Statistic:', result[0])
print('p-value:', result[1])
print('Critical Values:', result[4])                

plt.tight_layout()
plt.show()
