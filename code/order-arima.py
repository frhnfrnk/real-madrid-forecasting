import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('data/Real-Madrid-Statistic.csv', delimiter=";")
data['Year'] = pd.to_datetime(data['Year'], format='%Y')
data.set_index('Year', inplace=True)
data = data['Goal']  # Change to 'Goal' or 'Win'

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
plot_acf(data, ax=ax1, lags=20)
plot_pacf(data, ax=ax2, lags=8)  
plt.show()

p_values = range(0, 6)
d_values = [1] 
q_values = range(0, 6)

best_aic = np.inf
best_order = None

for p in p_values:
    for q in q_values:
        for d in d_values:
            order = (p, d, q)
            try:
                model = ARIMA(data, order=order)
                model_fit = model.fit()
                aic = model_fit.aic
                if aic < best_aic:
                    best_aic = aic
                    best_order = order
            except:
                continue

print('Best ARIMA Order (p, d, q):', best_order)
print('Best AIC:', best_aic)



# Result
# Points : Best ARIMA Order (p, d, q): (0, 1, 5)
# Goal : Best ARIMA Order (p, d, q): (1, 1, 0)
# Win : Best ARIMA Order (p, d, q): (1, 1, 0)
