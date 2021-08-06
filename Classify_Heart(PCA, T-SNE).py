import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier, plot_tree, plot_importance, XGBRFClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# cross_val_score (How accuracy it is)
# cv = cross_val_score(model7, X, y, cv=5, scoring='accuracy')
# score = np.mean(cv)
# print(score)

# Construct dataframe
df = pd.read_csv(r'../heart.csv', skip_blank_lines=True)
df = pd.DataFrame(df)
df = df.dropna()

features = ['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exng', 'oldpeak', 'slp', 'caa', 'thall'
            ]
dataset = df[features]
result = df['output']
y = result.values
X = df[features].values

# Corr.
M = df
correlated = M.corr()
sns.heatmap(correlated, annot=True)
plt.show()
# 'Output' has a obvious related to 'cp', 'thalachh', 'slp'

# Standardize or Normalize
scaler = StandardScaler()
X = scaler.fit_transform(X)

# scaler = MinMaxScaler()
# X = scaler.fit_transform(X)

# PCA
# X = pca.fit_transform(X)

# T-SNE (Less data)
T_SNE = TSNE(n_components=2)
X = T_SNE.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.7, random_state=77)

# General overview
sns.displot(x='age', hue='sex', data=df, alpha=0.6)
plt.title('Gender')
plt.show()

# Over fitting?
cv = cross_val_score(KNeighborsClassifier(), X, y, cv=10, scoring='accuracy')
score = np.mean(cv)
print('The cross-val score of Knn is : ', score)

cv = cross_val_score(RandomForestClassifier(), X, y, cv=10, scoring='accuracy')
score = np.mean(cv)
print('The cross-val score of RandomForest is : ', score)

cv = cross_val_score(DecisionTreeClassifier(), X, y, cv=10, scoring='accuracy')
score = np.mean(cv)
print('The cross-val score of DecisionTree is : ', score)

cv = cross_val_score(XGBRFClassifier(eval_metric='logloss', use_label_encoder=False), X, y, cv=10, scoring='accuracy')
score = np.mean(cv)
print('The cross-val score of xgboost is : ', score)

cv = cross_val_score(svm.SVC(), X, y, cv=10, scoring='accuracy')
score = np.mean(cv)
print('The cross-val score of SVM is : ', score)
#####
estimators = [('knn', KNeighborsClassifier()),
              ('rf', RandomForestClassifier()),
              ('dt', DecisionTreeClassifier()),
              ('svm', svm.SVC()),
              ('xgboost', XGBRFClassifier(eval_metric='logloss', use_label_encoder=False))]
stackingCl = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
stackingCl.fit(X_train, y_train)
######
cv = cross_val_score(stackingCl, X, y, cv=10, scoring='accuracy')
score = np.mean(cv)
print('The cross-val score of SVM is : ', score)
# Classify

xgboostModel = XGBRFClassifier(n_estimators=100, learning_rate=0.3)
xgboostModel.fit(X_train, y_train)
predicted = xgboostModel.predict(X_train)

print('The score XGBoost(train) is : ', xgboostModel.score(X_train, y_train))

df_trained = pd.DataFrame(X_train, columns=['PC1', 'PC2'])
df_trained['Index'] = y_train
df_trained['prediction'] = predicted

sns.lmplot('PC1', 'PC2', hue='Index', data=df_trained, fit_reg=False)
plt.title('XGBoost (train--real)')
plt.show()
sns.lmplot('PC1', 'PC2', hue='prediction', data=df_trained, fit_reg=False)
plt.title('XGBoost (train--prediction)')
plt.show()

# residplot
# sns.residplot(y_train, predicted)
# plt.show()

# test
predicted = xgboostModel.predict(X_test)

print('The score XGBoost(test) is : ', xgboostModel.score(X_test, y_test))

df_test = pd.DataFrame(X_test, columns=['PC1', 'PC2'])
df_test['Index'] = y_test
df_test['prediction'] = predicted

sns.lmplot('PC1', 'PC2', hue='Index', data=df_test, fit_reg=False)
plt.title('XGBoost (test--real)')
plt.show()
sns.lmplot('PC1', 'PC2', hue='prediction', data=df_test, fit_reg=False)
plt.title('XGBoost (test--prediction)')
plt.show()

# importance of XGboost
xgboostModel.fit(dataset, result)
plot_importance(xgboostModel)
plt.show()

# RandomForest
random_forestModel = RandomForestClassifier(n_estimators=100, criterion='gini')
random_forestModel.fit(X_train, y_train)
predicted = random_forestModel.predict(X_train)

print('The score random forest(train) is : ', random_forestModel.score(X_train, y_train))

df_trained = pd.DataFrame(X_train, columns=['PC1', 'PC2'])
df_trained['Index'] = y_train
df_trained['prediction'] = predicted

sns.lmplot('PC1', 'PC2', hue='Index', data=df_trained, fit_reg=False)
plt.title('Random Forest (train--real)')
plt.show()
sns.lmplot('PC1', 'PC2', hue='prediction', data=df_trained, fit_reg=False)
plt.title('Random Forest (train--prediction)')
plt.show()

# Test
predicted = random_forestModel.predict(X_test)

print('The score random forest(test) is : ', random_forestModel.score(X_test, y_test))

df_test = pd.DataFrame(X_test, columns=['PC1', 'PC2'])
df_test['Index'] = y_test
df_test['prediction'] = predicted

sns.lmplot('PC1', 'PC2', hue='Index', data=df_test, fit_reg=False)
plt.title('Random Forest (test--real)')
plt.show()
sns.lmplot('PC1', 'PC2', hue='prediction', data=df_test, fit_reg=False)
plt.title('Random Forest (test--prediction)')
plt.show()

# DecisionTree
decisiontreeModel = DecisionTreeClassifier(criterion='gini', max_depth=6)
decisiontreeModel.fit(X_train, y_train)
predicted = decisiontreeModel.predict(X_train)

df_trained = pd.DataFrame(X_train, columns=['PC1', 'PC2'])
df_trained['Index'] = y_train
df_trained['prediction'] = predicted

print('The score of decision tree(train) is : ', decisiontreeModel.score(X_train, y_train))
sns.lmplot('PC1', 'PC2', hue='Index', data=df_trained, fit_reg=False)
plt.title('Decision Tree (train--real)')
plt.show()
sns.lmplot('PC1', 'PC2', hue='prediction', data=df_trained, fit_reg=False)
plt.title('Decision Tree (train--prediction)')
plt.show()

# Test
predicted = decisiontreeModel.predict(X_test)

df_test = pd.DataFrame(X_test, columns=['PC1', 'PC2'])
df_test['Index'] = y_test
df_test['prediction'] = predicted

print('The score decision tree(test) is : ', decisiontreeModel.score(X_test, y_test))
sns.lmplot('PC1', 'PC2', hue='Index', data=df_test, fit_reg=False)
plt.title('Decision Tree (test--real)')
plt.show()
sns.lmplot('PC1', 'PC2', hue='prediction', data=df_test, fit_reg=False)
plt.title('Decision Tree (test--prediction)')
plt.show()

# SVM
# rbf
svcModel_rbf = svm.SVC(kernel='rbf', degree=3, gamma='auto', C=1)
svcModel_rbf.fit(X_train, y_train)
predicted = svcModel_rbf.predict(X_train)

accuracy = svcModel_rbf.score(X_train, y_train)
print('The accuracy of T-SNE method of svm_rbf is : ', accuracy)

df_trained = pd.DataFrame(X_train, columns=['PC1', 'PC2'])
df_trained['Index'] = y_train
df_trained['prediction'] = predicted

sns.lmplot('PC1', 'PC2', hue='Index', data=df_trained, fit_reg=False)
plt.title('Train--SVM')
plt.show()

sns.lmplot('PC1', 'PC2', hue='prediction', data=df_trained, fit_reg=False)
plt.title('Train--SVM (predict)')
plt.show()
# Tested
predicted = svcModel_rbf.predict(X_test)

accuracy = svcModel_rbf.score(X_test, y_test)
print('The accuracy of T-SNE method of svm_rbf(test) is : ', accuracy)

df_test = pd.DataFrame(X_test, columns=['PC1', 'PC2'])
df_test['Index'] = y_test
df_test['prediction'] = predicted

sns.lmplot('PC1', 'PC2', hue='Index', data=df_test, fit_reg=False)
plt.title('SVM--Test')
plt.show()

sns.lmplot('PC1', 'PC2', hue='prediction', data=df_test, fit_reg=False)
plt.title('SVM--Test (predict)')
plt.show()

# poly--result: 0.75 (rejected)
# svcModel_poly = svm.SVC(kernel='poly', degree=3, gamma='auto', C=1)

# Knn (Over fit?)
knnModel = KNeighborsClassifier(n_neighbors=5)

knnModel.fit(X_train, y_train)
predicted = knnModel.predict(X_train)

print('The score of knn trained is : ', knnModel.score(X_train, y_train))

df_trained = pd.DataFrame(X_train, columns=['PC1', 'PC2'])
df_trained['Index'] = y_train
df_trained['prediction'] = predicted

sns.lmplot('PC1', 'PC2', hue='Index', data=df_trained, fit_reg=False)
plt.title('Train--knn')
plt.show()

sns.lmplot('PC1', 'PC2', hue='prediction', data=df_trained, fit_reg=False)
plt.title('Train--knn (predict)')
plt.show()

# Test
predicted = knnModel.predict(X_test)

print('The score of knn test is : ', knnModel.score(X_test, y_test))

df_test = pd.DataFrame(X_test, columns=['PC1', 'PC2'])
df_test['Index'] = y_test
df_test['prediction'] = predicted

sns.lmplot('PC1', 'PC2', hue='Index', data=df_test, fit_reg=False)
plt.title('Test--knn')
plt.show()

sns.lmplot('PC1', 'PC2', hue='prediction', data=df_test, fit_reg=False)
plt.title('Test--knn (predict)')
plt.show()

# Stacking
estimators = [('knn', KNeighborsClassifier()),
              ('rf', RandomForestClassifier()),
              ('dt', DecisionTreeClassifier()),
              ('svm', svm.SVC()),
              ('xgboost', XGBRFClassifier(eval_metric='logloss', use_label_encoder=False))]
stackingCl = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
stackingCl.fit(X_train, y_train)

predicted = stackingCl.predict(X_train)

print('The score of stacking classify (train) is : ', stackingCl.score(X_train, y_train))

df_trained = pd.DataFrame(X_train, columns=['PC1', 'PC2'])
df_trained['Index'] = y_train
df_trained['prediction'] = predicted

sns.lmplot('PC1', 'PC2', hue='Index', data=df_trained, fit_reg=False)
plt.title('Train--stacking')
plt.show()

sns.lmplot('PC1', 'PC2', hue='prediction', data=df_trained, fit_reg=False)
plt.title('Train--stacking (predict)')
plt.show()

# test
predicted = stackingCl.predict(X_test)

print('The score of stacking test is : ', stackingCl.score(X_test, y_test))

df_test = pd.DataFrame(X_test, columns=['PC1', 'PC2'])
df_test['Index'] = y_test
df_test['prediction'] = predicted

sns.lmplot('PC1', 'PC2', hue='Index', data=df_test, fit_reg=False)
plt.title('Test--stacking')
plt.show()

sns.lmplot('PC1', 'PC2', hue='prediction', data=df_test, fit_reg=False)
plt.title('Test--syacking (predict)')
plt.show()

