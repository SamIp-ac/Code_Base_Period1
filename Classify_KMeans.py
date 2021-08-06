import pandas as pd
import math
import seaborn as sns
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from sklearn import metrics

# new
kMeanModel = KMeans(n_clusters=3, random_state=57)

df = pd.read_csv(r'../iot_telemetry_data.csv', skip_blank_lines=True)
df = pd.DataFrame(df)
df = df.dropna()
df = df.drop(columns=['ts', 'motion', 'light'])
print(df.shape)

label_map = {'b8:27:eb:bf:9d:51': 1, '00:0f:00:70:91:0a': 2, '1c:bf:ce:15:ec:4d': 3}
df['device'] = df['device'].map(label_map)
sns.lmplot('temp', 'humidity', hue='device', data=df, fit_reg=False)
plt.show()

data = df[['temp', 'humidity', 'device']]
dataset = data.drop(columns='device')
clusters_pre = kMeanModel.fit_predict(dataset)

print('The kMeanModel inertia is : ', kMeanModel.inertia_)

dataset['prediction'] = clusters_pre
sns.lmplot('temp', 'humidity', hue='prediction', data=dataset, fit_reg=False)

plt.scatter(kMeanModel.cluster_centers_[:, 0], kMeanModel.cluster_centers_[:, 1], s=200, c='r', marker='*')
plt.show()

kMeans_list = [KMeans(n_clusters=k).fit(df) for k in range(1, 10)]
inertias = [model.inertia_ for model in kMeans_list]

plt.plot(range(1, 10), inertias, 'bo-')
plt.show()

#######
tot = 0
for i in range(0, math.ceil((clusters_pre.shape[0] + 1)/10)):
    error = clusters_pre[i] - df['device'][i]
    tot = tot + error
print('The average error is : ', tot/math.ceil((clusters_pre.shape[0] + 1)/10))
rmse = metrics.mean_squared_error(clusters_pre, df['device'], squared=False)
print('The mean squared error is : ', rmse)
