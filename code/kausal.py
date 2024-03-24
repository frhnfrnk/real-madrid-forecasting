import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('data/Real-Madrid-Statistic.csv', delimiter=";")

data.set_index('Year', inplace=True)

X = data['Win'].values.reshape(-1, 1)
y = data['Points'].values

X_with_intercept = np.c_[np.ones(X.shape[0]), X]

coefficients = np.linalg.inv(X_with_intercept.T.dot(X_with_intercept)).dot(X_with_intercept.T).dot(y)

# Using mean of Win for 2023
forecasted_Win_2023 = data['Win'].mean()
forecasted_price_2023 = coefficients[0] + coefficients[1] * forecasted_Win_2023

print(f"The forecasted Win for the year 2023 is: {forecasted_Win_2023:.2f}")
print(f"The forecasted nickel price for the year 2023 is: ${forecasted_price_2023:.2f}")

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Original Data')
plt.plot(X, X_with_intercept.dot(coefficients), color='red', label='Regression Line')
plt.scatter(forecasted_Win_2023, forecasted_price_2023, color='green', label='Forecast for 2023')
plt.title('Nickel Prices and Linear Regression Forecasting')
plt.xlabel('Win')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)
plt.show()