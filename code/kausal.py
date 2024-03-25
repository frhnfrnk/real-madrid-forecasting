import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = pd.read_csv('data/Real-Madrid-Statistic.csv', delimiter=";")

X = data['Win'].values.reshape(-1, 1)
y = data['Points'].values

model = LinearRegression()
model.fit(X, y)

predict_value = np.array([[29]])
predicted_points = model.predict(predict_value)
print("Prediksi jumlah poin untuk tahun 2024:", predicted_points[0])

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, model.predict(X), color='red', label='Linear Regression Model')
plt.plot(predict_value, predicted_points, marker='o', markersize=8, color='green', label='Predicted Value: ' + str(round(predicted_points[0])))
plt.title('Linear Regression - Real Madrid Points Prediction')
plt.xlabel('Win')
plt.ylabel('Points')
plt.legend()
plt.grid(True)

plt.text(predict_value[0], predicted_points[0], str(round(predicted_points[0])), fontsize=12, ha='right')

plt.show()
