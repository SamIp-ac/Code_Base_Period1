import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics
from sklearn.svm import SVR
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import VotingRegressor
# For non-linear model, do the same procedure
# Knn
knnModel = KNeighborsRegressor(n_neighbors=5)

np.random.seed(71)
noise = np.random.rand(100, 1)
X = np.random.rand(100, 1)
y = 2*X + 43 + noise

plt.scatter(X, y, s=5, c='orange')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

knnModel.fit(X, y)
predicted = knnModel.predict(X)

print('The R2 score of knn is : ', knnModel.score(X, y))
rmse = metrics.mean_squared_error(y, predicted, squared=False)
print('The root mean squared of knn is : ', rmse)

plot1 = plt.figure(1)
plt.scatter(X, y, s=5, label='real', c='r')
plt.scatter(X, predicted, s=10, label='Prediction')
sns.regplot(x=X, y=predicted)
sns.regplot(x=X, y=y)
plt.title('Knn')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
# plt.show()

# residue
# sns.residplot(y, predicted)
# plt.show()

# SVR--linear
svrModel = SVR(C=1, kernel='linear')
svrModel.fit(X, y)
predicted = svrModel.predict(X)

print('The R2 score of SVR is : ', svrModel.score(X, y))
rmse = metrics.mean_squared_error(y, predicted, squared=False)
print('The root mean squared of SVR is : ', rmse)

plot2 = plt.figure(2)
plt.scatter(X, y, s=5, label='real', c='r')
plt.scatter(X, predicted, s=10, label='Prediction')
sns.regplot(x=X, y=predicted)
sns.regplot(x=X, y=y)
plt.legend()
plt.title('SVR-linear')
plt.xlabel('x')
plt.ylabel('y')

# SVR--poly
svrModel = SVR(C=1, kernel='poly', degree=2, gamma='auto')
svrModel.fit(X, y)
predicted = svrModel.predict(X)

print('The R2 score of SVR-poly is : ', svrModel.score(X, y))
rmse = metrics.mean_squared_error(y, predicted, squared=False)
print('The root mean squared of SVR-poly is : ', rmse)

plot3 = plt.figure(3)
plt.scatter(X, y, s=5, label='real', c='r')
plt.scatter(X, predicted, s=10, label='Prediction')
sns.regplot(x=X, y=y)
sns.regplot(x=X, y=predicted)
plt.legend()
plt.title('SVR-poly')
plt.xlabel('x')
plt.ylabel('y')

# SVR--rbf
svrModel = SVR(C=1, kernel='rbf', degree=2, gamma='auto')
svrModel.fit(X, y)
predicted = svrModel.predict(X)

print('The R2 score of SVR-rbf is : ', svrModel.score(X, y))
rmse = metrics.mean_squared_error(y, predicted, squared=False)
print('The root mean squared of SVR-rbf is : ', rmse)

plot4 = plt.figure(4)
plt.scatter(X, y, s=5, label='real', c='r')
plt.scatter(X, predicted, s=10, label='Prediction')
sns.regplot(x=X, y=y)
sns.regplot(x=X, y=predicted)
plt.legend()
plt.title('SVR-rbf')
plt.xlabel('x')
plt.ylabel('y')
plt.show()

# Decision Tree
decisiontreeModel = DecisionTreeRegressor(criterion='mse', max_depth=9, splitter='best')
decisiontreeModel.fit(X, y)
predicted = decisiontreeModel.predict(X)

print('The R2 score of Decision Tree is : ', decisiontreeModel.score(X, y))
rmse = metrics.mean_squared_error(y, predicted, squared=False)
print('The root mean squared of Decision Tree is : ', rmse)

plot5 = plt.figure(5)
plt.scatter(X, y, s=5, label='real', c='r')
plt.scatter(X, predicted, s=10, label='Prediction')
sns.regplot(x=X, y=y)
sns.regplot(x=X, y=predicted)
plt.legend()
plt.title('Decision Tree')
plt.xlabel('x')
plt.ylabel('y')


# Random Forest
randomforestModel = RandomForestRegressor(n_estimators=100, criterion='mse')
randomforestModel.fit(X, y)
predicted = randomforestModel.predict(X)

print('The R2 score of Random Forest is : ', randomforestModel.score(X, y))
rmse = metrics.mean_squared_error(y, predicted, squared=False)
print('The root mean squared of Random Forest is : ', rmse)

plot6 = plt.figure(6)
plt.scatter(X, y, s=5, label='real', c='orange')
plt.scatter(X, predicted, s=10, label='Prediction')
sns.regplot(x=X, y=y)
sns.regplot(x=X, y=predicted)
plt.legend()
plt.title('Random Forest')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
