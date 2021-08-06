import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier, plot_tree, plot_importance
from sklearn.linear_model import LogisticRegression

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
M = df[features]
correlated = M.corr()
sns.heatmap(correlated, annot=True)
plt.show()

# Standardize, Normalize
scaler = StandardScaler()
X = scaler.fit_transform(X)

# scaler = MinMaxScaler()
# X = scaler.fit_transform(X)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.7)
# , random_state=77

# Classify
# RandomForest
random_forestModel = RandomForestClassifier(n_estimators=100, criterion='gini')
random_forestModel.fit(X_train, y_train)
predicted = random_forestModel.predict(X_train)

print('The score random forest(train) is : ', random_forestModel.score(X_train, y_train))

df_trained = pd.DataFrame(X_train, columns=['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exng',
                                            'oldpeak', 'slp', 'caa', 'thall'])
df_trained['Index'] = y_train
df_trained['prediction'] = predicted

# Test
predicted = random_forestModel.predict(X_test)

print('The score random forest(test) is : ', random_forestModel.score(X_test, y_test))

df_test = pd.DataFrame(X_test, columns=['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exng',
                                        'oldpeak', 'slp', 'caa', 'thall'])
df_test['Index'] = y_test
df_test['prediction'] = predicted

# DecisionTree
decisiontreeModel = DecisionTreeClassifier(criterion='gini', max_depth=6)
decisiontreeModel.fit(X_train, y_train)
predicted = decisiontreeModel.predict(X_train)

df_trained = pd.DataFrame(X_train, columns=['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exng',
                                            'oldpeak', 'slp', 'caa', 'thall'])
df_trained['Index'] = y_train
df_trained['prediction'] = predicted

print('The score of decision tree(train) is : ', decisiontreeModel.score(X_train, y_train))

# Test
predicted = decisiontreeModel.predict(X_test)

df_test = pd.DataFrame(X_test, columns=['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exng',
                                        'oldpeak', 'slp', 'caa', 'thall'])
df_test['Index'] = y_test
df_test['prediction'] = predicted

print('The score decision tree(test) is : ', decisiontreeModel.score(X_test, y_test))

# SVM
# rbf
svcModel_rbf = svm.SVC(kernel='rbf', degree=3, gamma='auto', C=1)
svcModel_rbf.fit(X_train, y_train)
predicted = svcModel_rbf.predict(X_train)

accuracy = svcModel_rbf.score(X_train, y_train)
print('The accuracy of T-SNE method of svm_rbf(Train) is : ', accuracy)

df_trained = pd.DataFrame(X_train, columns=['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exng',
                                            'oldpeak', 'slp', 'caa', 'thall'])
df_trained['Index'] = y_train
df_trained['prediction'] = predicted

# Tested
predicted = svcModel_rbf.predict(X_test)

accuracy = svcModel_rbf.score(X_test, y_test)
print('The accuracy of T-SNE method of svm_rbf(Test) is : ', accuracy)

df_test = pd.DataFrame(X_test, columns=['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exng',
                                        'oldpeak', 'slp', 'caa', 'thall'])
df_test['Index'] = y_test
df_test['prediction'] = predicted

# poly--score result: 0.75 (rejected)
# svcModel_poly = svm.SVC(kernel='poly', degree=3, gamma='auto', C=1)

# XGBoost
xgboostModel = XGBClassifier(n_estimators=100, learning_rate=0.3)
xgboostModel.fit(X_train, y_train)
predicted = xgboostModel.predict(X_train)

print('The score XGBoost(train) is : ', xgboostModel.score(X_train, y_train))

df_trained = pd.DataFrame(X_train, columns=['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exng',
                                            'oldpeak', 'slp', 'caa', 'thall'])
df_trained['Index'] = y_train
df_trained['prediction'] = predicted

# test
predicted = xgboostModel.predict(X_test)

print('The score XGBoost(test) is : ', xgboostModel.score(X_test, y_test))

df_test = pd.DataFrame(X_test, columns=['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exng',
                                        'oldpeak', 'slp', 'caa', 'thall'])
df_test['Index'] = y_test
df_test['prediction'] = predicted

# importance of xgboost
xgboostModel.fit(dataset, result)
plot_importance(xgboostModel)
plt.title('Features importance of xgboost')
plt.show()
# Knn (Over fit?)
knnModel = KNeighborsClassifier(n_neighbors=5)

knnModel.fit(X_train, y_train)
predicted = knnModel.predict(X_train)

print('The score of knn trained is : ', knnModel.score(X_train, y_train))

df_trained = pd.DataFrame(X_train, columns=['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exng',
                                            'oldpeak', 'slp', 'caa', 'thall'])
df_trained['Index'] = y_train
df_trained['prediction'] = predicted

# Test
predicted = knnModel.predict(X_test)

print('The score of knn test is : ', knnModel.score(X_test, y_test))

df_test = pd.DataFrame(X_test, columns=['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exng',
                                        'oldpeak', 'slp', 'caa', 'thall'])
df_test['Index'] = y_test
df_test['prediction'] = predicted

# Stacking
estimators = [('knn', KNeighborsClassifier()),
              ('rf', RandomForestClassifier()),
              ('dt', DecisionTreeClassifier()),
              ('svm', svm.SVC())]
stackingCl = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
stackingCl.fit(X_train, y_train)

predicted = stackingCl.predict(X_train)

print('The score of stacking classify (train) is : ', stackingCl.score(X_train, y_train))

df_trained = pd.DataFrame(X_train, columns=['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exng',
                                            'oldpeak', 'slp', 'caa', 'thall'])
df_trained['Index'] = y_train
df_trained['prediction'] = predicted

# test
# Stacking
predicted = stackingCl.predict(X_test)

print('The score of stacking classify (test) is : ', stackingCl.score(X_test, y_test))

df_test = pd.DataFrame(X_test, columns=['age', 'sex', 'cp', 'trtbps', 'chol', 'fbs', 'restecg', 'thalachh', 'exng',
                                        'oldpeak', 'slp', 'caa', 'thall'])
df_test['Index'] = y_test
df_test['prediction'] = predicted
