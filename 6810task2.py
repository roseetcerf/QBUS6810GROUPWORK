# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 18:02:10 2017

@author: Administrator
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

df = pd.read_csv("Clothing_Store.csv")

# Get dummies
df = pd.get_dummies(df, drop_first=True)


X = df.iloc[:, :-1]
y = df.iloc[:,-1]

# Split the data into training and validation sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state = 460138105)




### Variable selection ###




# EDA
print(round(y_train.mean(), 3))

# Descriptive analysis
descriptive = X_train.describe()
descriptive.loc['skew', :] = X_train.skew()
descriptive.loc['kurt', :] = X_train.kurt()
print(descriptive.round(3))

# Histograms
import seaborn as sns
def hist(series):
    fig, ax = plt.subplots()
    sns.distplot(series, ax = ax, hist_kws = {'alpha': 0.9, 'edgecolor': 'black', 'color': sns.color_palette('Blues') [-1]},
                                              ked_kws = {'color': 'black', 'alpha':0.7})
    return fig, ax

# 必须先弄多少个predictor说
for i in range(50):
    hist(X_train[X[i]])
    sns.despine()
    plt.show()
    

for i in range(50):
    sns.regplot(X_train[X[i]], y_trian, color=sns.color_palette('Blues')[-1] ci = None, logistic=True, y_jitter = 0.05,
                scatter_kws={'s': 25, 'color':sns.color_palette('Blues')[-1], 'alpha': .5})
    sns.despine()
    plt.show()



# Predictor processing
# Logistic Regression
import statsmodels.api as sm
# use statasmodelto generate output and interpreting the model.
glm = sm.Logit(y_train, sm.add_constant(X_train)).fit()
print(glm.summary())


from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
# Fit and compute predictions
logit = LogisticRegression()
logit.fit(X_train, y_train)
y_pred = logit.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
logit_mis = 1- accuracy_score(y_test, y_pred)
print(logit_mis)


logit_l1 = LogisticRegressionCV(penalty = 'l1', solver = 'liblinear')
logit_l1.fit(X_train, y_train)
y_pred = logit_l1.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
logit_l1_mis = 1- accuracy_score(y_test, y_pred)
print(logit_l1_mis)


logit_l2 = LogisticRegressionCV(penalty = 'l2')
logit_l2.fit(X_train, y_train)
y_pred = logit_l2.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
logit_l2_mis = 1- accuracy_score(y_test, y_pred)
print(logit_l2_mis)



# KNN
from sklearn.model_selection import GridSearchCV
from sklearn import neighbors

k_grid = {"n_neighbors": np.arange(1, 32, 1)}
knn = GridSearchCV(neighbors.KNeighborsClassifier(), k_grid, cv = 5)
knn.fit(X_train, y_train)
knn_cv = knn.best_estimator_
print(knn_cv)

y_pred = knn.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
knn_mis = 1- accuracy_score(y_test, y_pred)
print(knn_mis)

# Decision Trees
from sklearn import tree

d_grid = {"max_depth": np.arange(2, 51, 1)}
tree = GridSearchCV(tree.DecisionTreeClassifier(), d_grid, cv = 5)
tree.fit(X_train, y_train)
tree_cv = tree.best_estimator_

y_pred = tree.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
tree_mis = 1- accuracy_score(y_test, y_pred)
print(tree_mis)


# Adaboost Classifier
from sklearn import ensemble

n_grid = {"n_estimators": np.arange(1, 101, 1)}
ada = GridSearchCV(ensemble.AdaBoostClassifier(algorithm='SAMME'), n_grid, cv = 5)
ada.fit(X_train, y_train)
ada_cv = ada.best_estimator_
print(best_esti)

y_pred = ada.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
ada_mis = 1- accuracy_score(y_test, y_pred)
print(ada_mis)



### Emsamble.




### ~N  Dscriminant analysis


### Model evaluation


