import pandas as pd
import time
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
svm = svm.LinearSVC()

list_ = pd.read_csv(r'../phone.csv', skip_blank_lines=True)
df = pd.DataFrame(list_)
df = df.dropna()
data = df[['battery_power', 'clock_speed', 'fc', 'int_memory', 'n_cores', 'pc', 'px_height', 'px_width', 'ram', 'sc_h',
           'talk_time', 'price_range']]
features = ['battery_power', 'clock_speed', 'fc', 'int_memory', 'n_cores', 'pc', 'px_height', 'px_width', 'ram', 'sc_h',
            'talk_time']

# data = data[data['price_range'] == [2 3]]
data2 = data[data['price_range'] == 2]
data3 = data[data['price_range'] == 3]
frames = [data2, data3]
data = pd.concat(frames)

print(df.columns)

X = data[['battery_power', 'clock_speed', 'fc', 'int_memory', 'n_cores', 'pc', 'px_height', 'px_width', 'ram', 'sc_h',
          'talk_time', 'price_range']].values
y = data['price_range'].values

X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.7, random_state=34)

# PCA
t_pca = time.time()
pca = PCA(n_components=2)
time_used_pca = time.time() - t_pca

PrincipalComponents = pca.fit_transform(X_train)

print('The variance ratio is : ', pca.explained_variance_ratio_)
print('The variance is : ', pca.explained_variance_)

plt.figure(figsize=(8, 6))
plt.scatter(PrincipalComponents[:, 0], PrincipalComponents[:, 1], s=1, c=y_train, alpha=1
            , cmap=plt.cm.get_cmap('Accent', 2))
plt.title('The PCA')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar()
plt.show()

pca_df = pd.DataFrame(data=PrincipalComponents, columns=['PC1', 'PC2'])

svm.fit(pca_df, y_train)
y_pred = svm.predict(pca_df)

accuracy = svm.score(pca_df, y_train)
print('The accuracy of PCA method of svm is : ', accuracy)

plot1 = plt.figure(1)
plt.scatter('PC1', 'PC2', data=pca_df, c=y_train, alpha=1, s=1)
plt.title('The actual')
plt.xlabel('PC1')
plt.ylabel('PC2')

plot2 = plt.figure(2)
plt.scatter('PC1', 'PC2', data=pca_df, c=y_pred, alpha=1, s=1)
plt.title('The prediction')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

# T-sne
T_SNE = TSNE(n_components=2)

t_t_sne = time.time()
PrincipalComponents = T_SNE.fit_transform(X_train)
time_used_TSNE = time.time() - t_t_sne

plt.figure(figsize=(8, 6))
plt.scatter(PrincipalComponents[:, 0], PrincipalComponents[:, 1], s=1, c=y_train, alpha=1
            , cmap=plt.cm.get_cmap('Accent', 2))
plt.title('The T-SNE')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar()
plt.show()

pca_df = pd.DataFrame(data=PrincipalComponents, columns=['PC1', 'PC2'])

svm.fit(pca_df, y_train)
y_pred = svm.predict(pca_df)

accuracy = svm.score(pca_df, y_train)
print('The accuracy of t-sne method of svm is : ', accuracy)

plot1 = plt.figure(1)
plt.scatter('PC1', 'PC2', data=pca_df, c=y_train, alpha=1, s=1)
plt.title('The actual')
plt.xlabel('PC1')
plt.ylabel('PC2')

plot2 = plt.figure(2)
plt.scatter('PC1', 'PC2', data=pca_df, c=y_pred, alpha=1, s=1)
plt.title('The prediction')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

print('The time used(pca) : ', time_used_pca)
print('The time used(T-sne) : ', time_used_TSNE)
print('The time ratio = ', time_used_TSNE/time_used_pca)

# Test

PrincipalComponents = T_SNE.fit_transform(X_test)

plt.figure(figsize=(8, 6))
plt.scatter(PrincipalComponents[:, 0], PrincipalComponents[:, 1], s=1, c=y_test, alpha=1
            , cmap=plt.cm.get_cmap('Accent', 2))
plt.title('The T-SNE')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar()
plt.show()

pca_df = pd.DataFrame(data=PrincipalComponents, columns=['PC1', 'PC2'])

svm.fit(pca_df, y_test)
y_pred = svm.predict(pca_df)

accuracy = svm.score(pca_df, y_test)
print('The accuracy of t-sne method of svm is : ', accuracy)

plot1 = plt.figure(1)
plt.scatter('PC1', 'PC2', data=pca_df, c=y_test, alpha=1, s=1)
plt.title('The actual')
plt.xlabel('PC1')
plt.ylabel('PC2')

plot2 = plt.figure(2)
plt.scatter('PC1', 'PC2', data=pca_df, c=y_pred, alpha=1, s=1)
plt.title('The prediction')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

# If we want to classify all 4 kinds of 'price range', we can use PCA -> knn
