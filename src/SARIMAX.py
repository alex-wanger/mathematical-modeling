import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
import numpy as np

# Load 2011 data for training
nyc_2011_df = pd.read_csv("nyc_gas_prices_2011.csv")
nyc_2011_df['Date'] = pd.to_datetime(nyc_2011_df['Date'])
nyc_2011_df.set_index('Date', inplace=True)
nyc_2011_df = nyc_2011_df.asfreq('W-MON')

# Load 2011-2012 data to get first week of 2012
nyc_2011_2012_df = pd.read_csv("nyc_gas_prices_2011_2012.csv")
nyc_2011_2012_df['Date'] = pd.to_datetime(nyc_2011_2012_df['Date'])
nyc_2011_2012_df.set_index('Date', inplace=True)
nyc_2011_2012_df = nyc_2011_2012_df.asfreq('W-MON')

# Get just the first value of 2012
nyc_2012_df = nyc_2011_2012_df[nyc_2011_2012_df.index.year == 2012]
first_2012_date = nyc_2012_df.index[0]
first_2012_actual = nyc_2012_df['NYC_Price'].iloc[0]

print(f"Training on: {len(nyc_2011_df)} observations from 2011")
print(f"Last training date: {nyc_2011_df.index[-1].strftime('%Y-%m-%d')}")
print(f"Predicting for: {first_2012_date.strftime('%Y-%m-%d')} (first week of 2012)")
print(f"Actual value: ${first_2012_actual:.4f}")
print("="*70)

# Build SARIMAX model with order (1, 2, 1)
print("\nBuilding SARIMAX(1, 2, 1) model...")

model = SARIMAX(nyc_2011_df['NYC_Price'],
                order=(1, 2, 1),
                enforce_stationarity=False,
                enforce_invertibility=False)

# Fit the model
model_fit = model.fit(disp=False, maxiter=500)

print(model_fit.summary())

# Forecast just 1 step ahead (first week of 2012)
print("\n" + "="*70)
print("Forecasting first week of 2012...")
print("="*70)

forecast = model_fit.get_forecast(steps=1)
predicted_value = forecast.predicted_mean.iloc[0]
conf_int = forecast.conf_int()

# Calculate error
error = first_2012_actual - predicted_value
abs_error = abs(error)
pct_error = (abs_error / first_2012_actual) * 100

print(f"\nPrediction Results:")
print(f"Date:             {first_2012_date.strftime('%Y-%m-%d')}")
print(f"Actual Value:     ${first_2012_actual:.4f}")
print(f"Predicted Value:  ${predicted_value:.4f}")
print(f"Error:            ${error:.4f}")
print(f"Absolute Error:   ${abs_error:.4f}")
print(f"Percent Error:    {pct_error:.2f}%")
print(f"95% CI:           [${conf_int.iloc[0, 0]:.4f}, ${conf_int.iloc[0, 1]:.4f}]")

# Check if actual value is within confidence interval
within_ci = conf_int.iloc[0, 0] <= first_2012_actual <= conf_int.iloc[0, 1]
print(f"Actual within CI: {'✓ Yes' if within_ci else '✗ No'}")

# Plot
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Plot 1: Full 2011 data plus prediction
axes[0].plot(nyc_2011_df.index, nyc_2011_df['NYC_Price'], 
             label='2011 Training Data', marker='o', color='blue', markersize=4)
axes[0].plot(first_2012_date, first_2012_actual, 
             label='Actual First Week 2012', marker='o', markersize=12, color='green')
axes[0].plot(first_2012_date, predicted_value, 
             label='Predicted First Week 2012', marker='x', markersize=12, color='red')
axes[0].errorbar(first_2012_date, predicted_value,
                yerr=[[predicted_value - conf_int.iloc[0, 0]], 
                      [conf_int.iloc[0, 1] - predicted_value]],
                fmt='none', color='red', alpha=0.3, capsize=8,
                label='95% Confidence Interval')
axes[0].axvline(x=nyc_2011_df.index[-1], color='gray', linestyle=':', 
                linewidth=2, label='End of Training')
axes[0].set_title('SARIMAX(1, 2, 1): Predicting First Week of 2012')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Price ($/gallon)')
axes[0].legend(loc='best')
axes[0].grid(True)

# Plot 2: Zoomed in on last 10 weeks of 2011 and first week of 2012
last_10_weeks = nyc_2011_df.iloc[-10:]
axes[1].plot(last_10_weeks.index, last_10_weeks['NYC_Price'], 
             label='Last 10 Weeks of 2011', marker='o', color='blue', markersize=6)
axes[1].plot(first_2012_date, first_2012_actual, 
             label='Actual First Week 2012', marker='o', markersize=12, color='green')
axes[1].plot(first_2012_date, predicted_value, 
             label='Predicted First Week 2012', marker='x', markersize=12, color='red')
axes[1].errorbar(first_2012_date, predicted_value,
                yerr=[[predicted_value - conf_int.iloc[0, 0]], 
                      [conf_int.iloc[0, 1] - predicted_value]],
                fmt='none', color='red', alpha=0.3, capsize=8)
axes[1].set_title('Zoomed View: Last 10 Weeks of 2011 + First Week of 2012')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Price ($/gallon)')
axes[1].legend(loc='best')
axes[1].grid(True)

plt.tight_layout()
plt.show()

# Show last few values of 2011 for context
print("\n" + "="*70)
print("Context - Last 5 weeks of 2011:")
print("="*70)
last_5 = nyc_2011_df.iloc[-5:]
for date, price in last_5['NYC_Price'].items():
    print(f"{date.strftime('%Y-%m-%d')}: ${price:.4f}")

print("\n→ Prediction for next week:")
print(f"{first_2012_date.strftime('%Y-%m-%d')}: ${predicted_value:.4f} (predicted)")
print(f"{first_2012_date.strftime('%Y-%m-%d')}: ${first_2012_actual:.4f} (actual)")