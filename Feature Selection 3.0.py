# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 15:18:53 2017

@author: Grace
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

train = pd.read_csv('Clean Train.csv')
y = pd.DataFrame(train.iloc[:, -1])
X = train.drop(['SalePrice', 'Unnamed: 0'], axis = 1) 

"""------- Standardize Data & Train-test Split ------"""

mu=X.mean()
sigma=X.std()
std_X=(X-mu)/sigma

std_X.fillna(0, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(std_X, y, random_state=1, train_size = 0.8)

"""------ LASSO CV ------"""
from sklearn.linear_model import LassoCV

lasso = LassoCV(cv = 5)

lasso.fit(X_train, np.ravel(y_train))

lasso_pred = lasso.predict(X_test)

lasso_mse = mean_squared_error(y_test, lasso_pred)

print("LASSO RMSE = {0:.3f}".format(np.sqrt(lasso_mse)))

columns = X.columns.values
lasso_cols = pd.Series(columns[np.nonzero(lasso.coef_)])
print("LASSO features: ", lasso_cols)

""" Refit the model with full training set """
X_full_test = pd.read_csv("Clean Test.csv")
X_full_test.drop('Unnamed: 0',  axis = 1, inplace = True)

lasso_pred_full = lasso.predict(X_full_test)
lasso_pred_full = pd.DataFrame(lasso_pred_full)
lasso_pred_full.to_csv('Submission 2.csv')






    
