import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np

# Load data
nyc_2011_df = pd.read_csv("nyc_gas_prices_2011.csv")
nyc_2011_df['Date'] = pd.to_datetime(nyc_2011_df['Date'])
nyc_2011_df.set_index('Date', inplace=True)
nyc_2011_df = nyc_2011_df.asfreq('W-MON')

nyc_2011_2012_df = pd.read_csv("nyc_gas_prices_2011_2012.csv")
nyc_2011_2012_df['Date'] = pd.to_datetime(nyc_2011_2012_df['Date'])
nyc_2011_2012_df.set_index('Date', inplace=True)
nyc_2011_2012_df = nyc_2011_2012_df.asfreq('W-MON')

nyc_2012_df = nyc_2011_2012_df[nyc_2011_2012_df.index.year == 2012]

print(f"Initial training: {len(nyc_2011_df)} observations from 2011")
print(f"Rolling forecast: {len(nyc_2012_df)} observations in 2012")
print("="*70)

# Test multiple model specifications with rolling forecast
model_specs = [
    ((1,0,1), "ARIMA(1,0,1) - No differencing"),
    ((2,0,1), "ARIMA(2,0,1) - No differencing"),
    ((1,1,1), "ARIMA(1,1,1) - First difference"),
    ((2,1,2), "ARIMA(2,1,2) - First difference"),
]

all_results = {}

for order, description in model_specs:
    print(f"\n{description}...")
    rolling_predictions = []
    training_data = nyc_2011_df['NYC_Price'].copy()
    
    for i in range(len(nyc_2012_df)):
        try:
            model = SARIMAX(training_data, 
                            order=order,
                            enforce_stationarity=False,
                            enforce_invertibility=False)
            
            model_fit = model.fit(disp=False, maxiter=200, method='lbfgs')
            forecast = model_fit.forecast(steps=1)
            rolling_predictions.append(forecast.iloc[0])
            
            actual_value = nyc_2012_df['NYC_Price'].iloc[i]
            training_data = pd.concat([training_data, pd.Series([actual_value], 
                                                                 index=[nyc_2012_df.index[i]])])
        except Exception as e:
            rolling_predictions.append(np.nan)
    
    rolling_series = pd.Series(rolling_predictions, index=nyc_2012_df.index)
    mape = np.mean(np.abs((nyc_2012_df['NYC_Price'] - rolling_series) / nyc_2012_df['NYC_Price'])) * 100
    
    all_results[description] = {
        'series': rolling_series,
        'mape': mape,
        'order': order
    }
    
    print(f"  MAPE: {mape:.2f}%")

# Plot all models
plt.figure(figsize=(14, 8))
plt.plot(nyc_2011_df.index, nyc_2011_df['NYC_Price'], 
         label='2011 Training', marker='o', color='blue', markersize=3, alpha=0.5)
plt.plot(nyc_2012_df.index, nyc_2012_df['NYC_Price'], 
         label='2012 Actual', marker='o', color='green', markersize=5, linewidth=3)

colors = ['red', 'orange', 'purple', 'brown']
linestyles = ['--', '-.', ':', '--']

for i, (desc, result) in enumerate(all_results.items()):
    plt.plot(result['series'].index, result['series'], 
             label=f"{desc} (MAPE={result['mape']:.1f}%)",
             linestyle=linestyles[i], color=colors[i], linewidth=2, marker='x', markersize=4)

plt.axvline(x=nyc_2011_df.index[-1], color='gray', linestyle=':', linewidth=2)
plt.title('Rolling Forecast Comparison: Different ARIMA Models')
plt.xlabel('Date')
plt.ylabel('Price ($/gallon)')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.show()

# Zoomed view - 2012 only
plt.figure(figsize=(14, 6))
plt.plot(nyc_2012_df.index, nyc_2012_df['NYC_Price'], 
         label='2012 Actual', marker='o', color='green', markersize=6, linewidth=3)

for i, (desc, result) in enumerate(all_results.items()):
    plt.plot(result['series'].index, result['series'], 
             label=f"{desc} (MAPE={result['mape']:.1f}%)",
             linestyle=linestyles[i], color=colors[i], linewidth=2, marker='x', markersize=4)

plt.title('2012 Rolling Forecasts: Model Comparison')
plt.xlabel('Date')
plt.ylabel('Price ($/gallon)')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.show()

print("\n" + "="*70)
print("Summary:")
sorted_results = sorted(all_results.items(), key=lambda x: x[1]['mape'])
for desc, result in sorted_results:
    print(f"{desc}: MAPE = {result['mape']:.2f}%")