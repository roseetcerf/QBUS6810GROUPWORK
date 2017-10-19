# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 22:09:14 2017

@author: Grace
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, RidgeCV, Lasso, ElasticNetCV, LassoCV
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import time

X = pd.read_csv('Standardized X.csv')
y = pd.DataFrame(pd.read_csv('Clean train.csv').iloc[:, -1])
std_X_test = pd.read_csv('Standardized test set.csv')

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, train_size = 0.8)

"""------ Grid Search ------"""
a_grid = {"alpha": np.arange(530, 540, 2)}

lasso_cv = GridSearchCV(Lasso(), a_grid, scoring='neg_mean_squared_error',cv = 50)

print("Running...")
tic = time.time()
lasso_cv.fit(X, y)

toc = time.time()
print("Training time: {0}s".format(toc - tic))

best_k = lasso_cv.best_estimator_
print(best_k)
print(np.sqrt(np.absolute(lasso_cv.best_score_)))

columns = X.columns.values
lasso_cols = pd.Series(columns[np.nonzero(best_k.coef_)])

"""------ Lasso CV ------"""

las = LassoCV(cv = 50)
las.fit(X_train, np.ravel(y_train))
las_pred = las.predict(X_test)
print(np.sqrt(mean_squared_error(y_test, las_pred)))

las_cols = pd.Series(columns[np.nonzero(las.coef_)])

las.fit(X, np.ravel(y))
las_full_pred = las.predict(std_X_test)
pd.Series(las_full_pred).to_csv('Submission 12.csv')




