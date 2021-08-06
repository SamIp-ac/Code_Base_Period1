import pandas as pd
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.svm import LinearSVC

scaler = StandardScaler()
svcModel = svm.LinearSVC(C=1, loss="hinge", random_state=33)

list_ = pd.read_csv(r'../phone.csv', skip_blank_lines=True)
df = pd.DataFrame(list_)
df = df.dropna()
features = ['battery_power', 'clock_speed', 'fc', 'int_memory', 'n_cores', 'pc', 'px_height', 'px_width', 'ram', 'sc_h',
            'talk_time', 'price_range']

M = df[features]
correlated = M.corr()
print(correlated)
sns.heatmap(correlated)
plt.show()

data = df[['battery_power', 'ram', 'price_range']]

data1 = data[data['price_range'] == 1]
data2 = data[data['price_range'] == 2]
data3 = data[data['price_range'] == 3]
frames = [data1, data2, data3]
data = pd.concat(frames)

sns.lmplot('battery_power', 'ram', hue='price_range', data=data, fit_reg=False)
plt.show()

X = data[['battery_power', 'ram']]
y = data['price_range']

svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("linear_svc", svcModel)])

svm_clf.fit(X, y)
prediction = svm_clf.predict(X)
data['prediction'] = prediction
sns.lmplot('battery_power', 'ram', hue='prediction', data=data, fit_reg=False)
plt.show()

print(svcModel.coef_)

print(svcModel.intercept_)

##### poly
svcModel_poly = svm.SVC(kernel='poly', degree=3, gamma='auto', C=1)

svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("linear_svc", svcModel_poly)])

svm_clf.fit(X, y)
prediction = svm_clf.predict(X)
data['prediction'] = prediction
sns.lmplot('battery_power', 'ram', hue='prediction', data=data, fit_reg=False)
plt.show()

##### RBF
svcModel_rbf = svm.SVC(kernel='rbf', degree=3, gamma='auto', C=1)

svm_clf = Pipeline([
        ("scaler", StandardScaler()),
        ("linear_svc", svcModel_rbf)])

svm_clf.fit(X, y)
prediction = svm_clf.predict(X)
data['prediction'] = prediction
sns.lmplot('battery_power', 'ram', hue='prediction', data=data, fit_reg=False)
plt.show()
