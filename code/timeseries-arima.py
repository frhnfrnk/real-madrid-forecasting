import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

data = pd.read_csv('data/Real-Madrid-Statistic.csv', delimiter=";")
data['Year'] = pd.to_datetime(data['Year'], format='%Y')
data.set_index('Year', inplace=True)
data = data['Points']

# Points : Best ARIMA Order (p, d, q): (0, 1, 5)
# Goal : Best ARIMA Order (p, d, q): (1, 1, 0)
# Win : Best ARIMA Order (p, d, q): (1, 1, 0)

#Points
order = (0, 1, 5)
model = ARIMA(data, order=order)
model_fit = model.fit()

forecast_next_3_years = model_fit.forecast(steps=3)
print('Forecast for the next 3 years:')
print(forecast_next_3_years)

plt.figure(figsize=(10, 6))
plt.plot(data.index, data, label='Actual')
plt.plot(forecast_next_3_years.index, forecast_next_3_years, marker='o', markersize=5, color='red', label='Forecast')
plt.title('Actual vs Forecasted Points')
plt.xlabel('Year')
plt.ylabel('Points')
plt.legend()

for i, point in enumerate(forecast_next_3_years):
        plt.text(forecast_next_3_years.index[i], point, str(round(point)), ha='right', va='bottom')

plt.grid(True)
plt.show()
