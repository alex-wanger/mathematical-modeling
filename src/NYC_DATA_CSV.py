import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../data/GAS_DATA.csv", header=2)

df.columns = df.columns.str.strip()

nyc_col = [col for col in df.columns if 'New York City' in col][0]

nyc_df = df[['Date', nyc_col]].copy()
nyc_df = nyc_df.rename(columns={nyc_col: 'NYC_Price'})

nyc_df['Date'] = pd.to_datetime(nyc_df['Date'], errors='coerce')

# fuckass python uses or
nyc_2011_df = nyc_df[nyc_df['Date'].dt.year == 2011]
#nyc_2011_df = nyc_df[(nyc_df['Date'] >= '2011-01-01') & (nyc_df['Date'] < '2013-01-01')]

nyc_2011_df.to_csv("nyc_gas_prices_2011_2012.csv", index=False)
print("Saved nyc_gas_prices_2011.csv")

plt.figure(figsize=(10,5))
plt.plot(nyc_2011_df['Date'], nyc_2011_df['NYC_Price'], marker='o')
plt.title("NYC Gas Prices 2011")
plt.xlabel("Date")
plt.ylabel("Price ($/gallon)")
plt.grid(True)
plt.show()


