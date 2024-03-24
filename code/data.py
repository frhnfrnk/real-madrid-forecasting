import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data/Real-Madrid-Statistic.csv', delimiter=";")

data['Year'] = pd.to_datetime(data['Year'], format='%Y')
data.set_index('Year', inplace=True)
data.sort_index(inplace=True)

plt.figure(figsize=(10, 6))
# plt.plot(data.index, data['Points'], label='Points', color='blue',marker='o', linestyle='-')
# plt.plot(data.index, data['Goal'], label='Goals', color='red', marker='o', linestyle='-')
plt.plot(data.index, data['Win'], label='Win', color='crimson', marker='o', linestyle='-')
plt.title('Real Madrid Win Over Time')
plt.xlabel('Year')
# plt.ylabel('Points')
# plt.ylabel('Goal')
plt.ylabel('Win')
plt.grid(True)
plt.show()
