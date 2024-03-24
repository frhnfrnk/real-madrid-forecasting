import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load data from CSV
df = pd.read_csv('data/Real-Madrid-Statistic.csv', delimiter=";")

# Preprocess data
df['Year'] = pd.to_datetime(df['Year'])
df.set_index('Year', inplace=True)
df = df.astype(int)

# Define a function for forecasting using rolling mean and rolling standard deviation
def forecast_rolling_mean_std(data, steps):
    rolling_mean = data.rolling(window=5).mean()
    rolling_std = data.rolling(window=5).std()
    last_value = data.iloc[-1]
    forecast = [last_value]

    for i in range(steps):
        next_value = np.random.normal(rolling_mean.iloc[-1], rolling_std.iloc[-1])
        forecast.append(next_value)

    return forecast

# Perform forecasting
steps = 5  # Change this according to your needs
forecast_win = forecast_rolling_mean_std(df['Win'], steps)
forecast_goal = forecast_rolling_mean_std(df['Goal'], steps)
forecast_points = forecast_rolling_mean_std(df['Points'], steps)

# Plotting

# Print forecasted values
print("Forecasted Wins:", forecast_win[1:])
print("Forecasted Goals:", forecast_goal[1:])
print("Forecasted Points:", forecast_points[1:])
