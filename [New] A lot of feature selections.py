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

# Standardize the test set predictors
mu_test = X_full_test.mean()

sigma_test = X_full_test.std()

std_X_full = (X_full_test-mu_test)/sigma_test

std_X_full.fillna(0, inplace=True)

# Predict with Standardized test set
lasso_pred_full = lasso.predict(std_X_full)
lasso_pred_full = pd.DataFrame(lasso_pred_full)
lasso_pred_full.to_csv('Submission 3.csv')


"""------ Elastic Net CV ------"""
from sklearn.linear_model import ElasticNetCV

elas = ElasticNetCV(cv = 500)

elas.fit(X_train, np.ravel(y_train))

elas_pred = elas.predict(X_test)

elas_rmse = np.sqrt(mean_squared_error(y_test, elas_pred))

print("ElasticNet RMSE = {0:.3f}".format(elas_rmse))

elas_pred_full = elas.predict(std_X_full)
elas_pred_full = pd.DataFrame(elas_pred_full)
elas_pred_full.to_csv('Submission 8.csv')









#"""------ Forward Selection ------"""
#
#from statlearning import forward
#
#fwd = forward()
#fwd.fit(X_train, y_train)
#
#fwdsubset = fwd.subset
#
## Select variables that are chosen by forward
#
#trans_X = X_train.T
#
#new_X = []
#
#for i in fwdsubset:
#    new_X.append(trans_X.iloc[i, :])
#
#new_X = pd.DataFrame(new_X).T
#                    
#"""------ Forward Prediction ------"""
#import time
#
#tic = time.time()
#
#fwd_pred = fwd.predict(X_test)
#
#toc = time.time()
#
#print("Time: {0}s".format(toc - tic))
#
#fwd_mse = mean_squared_error(y_test, fwd_pred)
#print("Forward Selection RMSE = {0:.3F}".format(np.sqrt(fwd_mse)))
#
#fwd_pred_full = pd.DataFrame(fwd.predict(std_X_full), columns = {'Prediction'})
#fwd_pred_full.to_csv('Submission 4.csv')
#
#"""------ LASSOCV on Forward selected features ------"""
#lasso_2 = LassoCV(cv = 5)
#
#lasso_2.fit(X_train, y_train)    
#
#lasso_2_pred = lasso_2.predict(X_test)
#
#lasso_2_mse = mean_squared_error(y_test, lasso_2_pred)
#print("LASSO 2 RMSE = {0:.3F}".format(np.sqrt(lasso_2_mse)))
#
#"""------ PCA ------"""
#from sklearn.decomposition import PCA
#from sklearn.model_selection import cross_val_score
#from sklearn.model_selection import KFold
#
#from statlearning import pcrCV
#
#pcrcv = pcrCV(X_train, y_train)
#
#pcrcv.fit(X_train, y_train)

pcrcv_pred = pcrcv.predict(X_test)

pcrcv_rmse = np.sqrt(mean_squared_error(y_test, pcrcv_pred))

"""------ PLS ------"""
from statlearning import plsCV

plscv = plsCV(X_train, y_train)

#pcrcv.fit(X_train, y_train)

plscv_pred = plscv.predict(X_test)

plscv_rmse = np.sqrt(mean_squared_error(y_test, plscv_pred))

"""------ Huber Loss Regressor ------"""
from sklearn.linear_model import HuberRegressor

hbr = HuberRegressor()

hbr.fit(X_train, y_train)

hbr_pred = hbr.predict(X_test)

hbr_rmse = np.sqrt(mean_squared_error(y_test, hbr_pred))
print('Huber Regresor RMSE = {0:.3f}'.format(hbr_rmse))

hbr_pred_full = hbr.predict(std_X_full)
