import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Gender : Man
scaler = StandardScaler()
pca = PCA(n_components=2)

knnModel = KNeighborsClassifier(n_neighbors=3)

list_ = pd.read_csv(r'../500_Person_Gender_Height_Weight_Index.csv', skip_blank_lines=True)
df = pd.DataFrame(list_)
df = df.dropna()

data_Man = df[df['Gender'] == 'Male']
data_Man = data_Man.drop(columns='Gender')

X = data_Man[['Height', 'Weight']].values
y = data_Man['Index'].values

print('The empty entries of X is : ', len(np.where(np.isnan(X))[0]))
print('The empty entries of y is : ', len(np.where(np.isnan(y))[0]))

# StandardScaler(), PCA
# X = scaler.fit_transform(X)
# X = pca.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.7, random_state=53)

knnModel.fit(X_train, y_train)
predicted = knnModel.predict(X_train)

print('The score trained is : ', knnModel.score(X_train, y_train))

df_trained = pd.DataFrame(X_train, columns=['Height', 'Weight'])
df_trained['Index'] = y_train

df_trained['prediction'] = predicted

sns.lmplot('Height', 'Weight', hue='Index', data=df_trained, fit_reg=False)
plt.show()

sns.lmplot('Height', 'Weight', hue='prediction', data=df_trained, fit_reg=False)
plt.show()

# tested
df_tested = pd.DataFrame(X_test, columns=['Height', 'Weight'])
df_tested['Index'] = y_test

knnModel.fit(X_test, y_test)
predicted = knnModel.predict(X_test)

df_tested['prediction'] = predicted

# Sometimes may not exist the tested 'index'
sns.lmplot('Height', 'Weight', hue='Index', data=df_tested, fit_reg=False)
plt.show()

sns.lmplot('Height', 'Weight', hue='prediction', data=df_tested, fit_reg=False)
plt.show()

print('The score of tested is : ', knnModel.score(X_test, y_test))
