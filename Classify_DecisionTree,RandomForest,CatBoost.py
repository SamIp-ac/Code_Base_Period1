import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import metrics
from catboost import CatBoostClassifier
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.decomposition import PCA
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier

scaler = StandardScaler()
pca = PCA(n_components=2)
decisiontreeModel = DecisionTreeClassifier(criterion='gini', max_depth=6, random_state=83)

list_ = pd.read_csv(r'../500_Person_Gender_Height_Weight_Index.csv', skip_blank_lines=True)
df = pd.DataFrame(list_)
df = df.dropna()

data_Man = df[df['Gender'] == 'Male']
data_Man = data_Man.drop(columns='Gender')
X = data_Man[['Height', 'Weight']]
y = data_Man['Index'].values

print('The empty entries of X is : ', len(np.where(np.isnan(X))[0]))
print('The empty entries of y is : ', len(np.where(np.isnan(y))[0]))

# PCA switch
# StandardScaler(), PCA
# X = scaler.fit_transform(X)
# X = pca.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.7)

decisiontreeModel.fit(X_train, y_train)
prediction = decisiontreeModel.predict(X_train)

df_trained = pd.DataFrame(X_train, columns=['Height', 'Weight'])
df_trained['Index'] = y_train
df_trained['prediction'] = prediction

print('The score of decision tree(train) is : ', decisiontreeModel.score(X_train, y_train))
sns.lmplot('Height', 'Weight', hue='Index', data=df_trained, fit_reg=False)
plt.title('Decision Tree (train--real)')
plt.show()
sns.lmplot('Height', 'Weight', hue='prediction', data=df_trained, fit_reg=False)
plt.title('Decision Tree (train--prediction)')
plt.show()

prediction = decisiontreeModel.predict(X_test)

df_test = pd.DataFrame(X_test, columns=['Height', 'Weight'])
df_test['Index'] = y_test
df_test['prediction'] = prediction

print('The score decision tree(test) is : ', decisiontreeModel.score(X_test, y_test))
sns.lmplot('Height', 'Weight', hue='Index', data=df_test, fit_reg=False)
plt.title('Decision Tree (test--real)')
plt.show()
sns.lmplot('Height', 'Weight', hue='prediction', data=df_test, fit_reg=False)
plt.title('Decision Tree (test--prediction)')
plt.show()

# Visualization
# dot_data = export_graphviz(decisiontreeModel, out_file=None, feature_names=['Height', 'Weight'],
                           # class_names=['0', '1', '2', '3', '4', '5'],
                           # filled=True, rounded=True,
                           # special_characters=True)
# graph = graphviz.Source(dot_data, format="png")
# print(graph)
# graph.render('decision_tree_graphivz')
# 'decision_tree_graphivz.png'

# 2nd method
# fig = plt.figure(figsize=(50, 40))
# _ = tree.plot_tree(decisiontreeModel,
#                    feature_names=['Height', 'Weight'],
#                    class_names=['0', '1', '2', '3', '4', '5'],
#                    filled=True, rounded=True)
# fig.savefig('decistion_tree.jpg')

# Random forest

random_forestModel = RandomForestClassifier(n_estimators=100, criterion='gini')
random_forestModel.fit(X_train, y_train)
prediction = random_forestModel.predict(X_train)

print('The score random forest(train) is : ', random_forestModel.score(X_train, y_train))

df_trained = pd.DataFrame(X_train, columns=['Height', 'Weight'])
df_trained['Index'] = y_train
df_trained['prediction'] = prediction

sns.lmplot('Height', 'Weight', hue='Index', data=df_trained, fit_reg=False)
plt.title('Random Forest (train--real)')
plt.show()
sns.lmplot('Height', 'Weight', hue='prediction', data=df_trained, fit_reg=False)
plt.title('Random Forest (train--prediction)')
plt.show()

prediction = random_forestModel.predict(X_test)
print('The score random forest(test) is : ', decisiontreeModel.score(X_test, y_test))

df_test = pd.DataFrame(X_test, columns=['Height', 'Weight'])
df_test['Index'] = y_test
df_test['prediction'] = prediction

sns.lmplot('Height', 'Weight', hue='Index', data=df_test, fit_reg=False)
plt.title('Random Forest (test--real)')
plt.show()
sns.lmplot('Height', 'Weight', hue='prediction', data=df_test, fit_reg=False)
plt.title('Random Forest (test--prediction)')
plt.show()

print('The importance features : ', random_forestModel.feature_importances_)
# Visualization
# fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10, 2), dpi=900)
# for index in range(0, 3):
#     tree.plot_tree(random_forestModel.estimators_[index], feature_names=['Height', 'Weight'],
#                    class_names=['0', '1', '2', '3', '4', '5'],
#                    filled=True, rounded=True, ax=axes[index])
# fig.savefig('decistion_tree_random_forest.jpg')

# Catboost
model = CatBoostClassifier(verbose=0)
model.fit(X_train, y_train)

predicted0 = model.predict(X_test)

df_test = pd.DataFrame(X_test, columns=['Height', 'Weight'])
df_test['Index'] = y_test
df_test['prediction'] = predicted0

print('The score of catboost (test) is : ', model.score(X_test, y_test))
sns.lmplot('Height', 'Weight', hue='Index', data=df_test, fit_reg=False)
plt.title('Catboost (test--real)')
plt.show()
sns.lmplot('Height', 'Weight', hue='prediction', data=df_test, fit_reg=False)
plt.title('Catboost (test--prediction)')
plt.show()

# accuracy score
print('-----------------classification_report-------------------')
print(metrics.classification_report(y_test, predicted0))
print('jaccard_similarity_score', metrics.jaccard_score(y_test, predicted0))
print('log_loss', metrics.log_loss(y_test, predicted0))
print('zero_one_loss', metrics.zero_one_loss(y_test, predicted0))
print('AUC&ROC', metrics.roc_auc_score(y_test, predicted0))
print('matthews_corrcoef', metrics.matthews_corrcoef(y_test, predicted0))

sns.heatmap(metrics.confusion_matrix(y_test, predicted0), annot=True)
plt.title("Confusion Matrix 0")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
