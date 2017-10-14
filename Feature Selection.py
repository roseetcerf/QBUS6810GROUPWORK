# -*- coding: utf-8 -*-
"""
Created on Sun Sep 24 15:08:35 2017

@author: Grace
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error

X = pd.read_csv('X.csv')
y = pd.read_csv('y.csv')
data = pd.concat([y, X], axis =1)


print(y.describe())
correlation = data.corr().SalePrice
                       
X.fillna(0, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, train_size = 0.8)

""" Feature selection: LASSO """

lasso = LassoCV(cv = 5)
# Fit training set
lasso.fit(X_train, np.ravel(y_train))
# Predict on test set
lasso_pred = lasso.predict(X_test)
# Compute R-squared score and MSE
lasso_mse = mean_squared_error(y_test, lasso_pred)

print(np.sqrt(lasso_mse))

columns = X.columns.values
lasso_cols = pd.Series(columns[np.nonzero(lasso.coef_)])
print("LASSO features: ",lasso_cols)

""" Feature selection: Elastic Net """
elascv = ElasticNetCV(cv = 5)
# Fit training set
elascv.fit(X_train, np.ravel(y_train))
# Predict on test set
elascv_pred = elascv.predict(X_test)
# Compute mse
elas_mse = mean_squared_error(y_test, elascv_pred)

print(np.sqrt(elas_mse))

elas_cols = pd.Series(columns[np.nonzero(elascv.coef_)])
print("Elastic Net features: ", elas_cols)
