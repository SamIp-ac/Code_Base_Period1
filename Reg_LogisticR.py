import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

LogisticModel = LogisticRegression(random_state=0)
MeansModel = KMeans(n_clusters=3)

lis = pd.read_csv(r'../SleepStudyData.csv', skip_blank_lines=True)

df = pd.DataFrame(lis)
df = df.replace('Yes,', 'Yes')
df = df.replace('No,', 'No')
df = df.dropna()

data = pd.DataFrame()
data = df.filter(['Hours'])
data = data.dropna()

label_map = {'No': 0, 'Yes': 1}
# Enough,Hours,PhoneReach,PhoneTime,Tired,Breakfast
data['Tired'] = df['Tired']
data['Class_Enough'] = df['Enough'].map(label_map)
data['Class_PhoneReach'] = df['PhoneReach'].map(label_map)
data['Class_PhoneTime'] = df['PhoneTime'].map(label_map)
data['Breakfast'] = df['Breakfast'].map(label_map)

print(np.where(np.isnan(data)))
print(data)

data0 = data[['Hours', 'Tired']]
data1 = data['Class_Enough'].values

X_train, X_test, y_train, y_test = train_test_split(data0, data1, train_size=.7, random_state=12)

LogisticModel.fit(X_train, y_train)
l1 = LogisticModel.score(X_train, y_train)
l2 = LogisticModel.score(X_test, y_test)

df_train = pd.DataFrame(X_train)
df_train['Class_Enough'] = y_train

predict = LogisticModel.predict(X_test)
df_test = pd.DataFrame(X_test)
df_test['Class_Enough'] = y_test
df_test['Prediction'] = predict

plot1 = plt.figure(1)
sns.lmplot('Hours', 'Tired', hue='Prediction', data=df_test, fit_reg=False)
plt.title('Prediction')

plot2 = plt.figure(2)
sns.lmplot('Hours', 'Tired', hue='Class_Enough', data=df_test, fit_reg=False)
plt.title('Test')

plot3 = plt.figure(3)
sns.lmplot('Hours', 'Tired', hue='Class_Enough', data=data, fit_reg=False)
plt.title('Actual')
plt.show()

print('The score of train sample is : ', l1)
print('The score of test sample is : ', l2)
