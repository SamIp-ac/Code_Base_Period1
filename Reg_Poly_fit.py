import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

df = pd.read_csv(r'../0823.HK.csv', skip_blank_lines=True)
df = df.filter(['Close'])
data = df.dropna()
dataset = data.values
dataset = np.array(dataset)
plt.plot(dataset, label='Actual values')

t = np.array(range(0, dataset.shape[0]))
t = t.reshape(-1, 1)

f = np.polyfit(t[0:, 0], dataset[0:, 0], 12)
p = np.poly1d(f)
print(p)

y_val = p(t)

# plt.plot(t, dataset, '.', label='original values')
plt.plot(t, y_val, 'r', label='polyfit. values')
plt.xlabel('Days')
plt.ylabel('Closing Price')
plt.legend(loc='best')
plt.title('poly fitting')
plt.show()

# Polynomial Features
pf = PolynomialFeatures(degree=12, include_bias=False)

df = pd.read_csv(r'../0823.HK.csv', skip_blank_lines=True)
df = df['Close']
data = df.dropna()
dataset = data.values

t0 = np.array(range(0, dataset.shape[0]))
t0 = t0.reshape(-1, 1)

X_poly = pf.fit_transform(t0)
print(X_poly.shape)
print(dataset.shape)

lin_reg = LinearRegression()
lin_reg.fit(X_poly, dataset)
print(lin_reg.intercept_, lin_reg.coef_)

X_new = np.array(range(0, dataset.shape[0] + 100))
# Prediction : X_new = np.array(range(0, dataset.shape[0] + 1000))
X_new = X_new.reshape(-1, 1)
X_new_poly = pf.transform(X_new)
y_new = lin_reg.predict(X_new_poly)

plt.plot(t0, dataset, 'b', label='Actual values')
plt.plot(X_new, y_new, 'r', linewidth=2, label='Poly features')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc='best')
plt.show()
