# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 15:18:53 2017

@author: Grace
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

tic = time.time()

train = pd.read_csv('Clean Train.csv')
y = pd.DataFrame(train.iloc[:, -1])
X = train.drop(['SalePrice', 'Unnamed: 0'], axis = 1) 

"""------- Standardize Data & Train-test Split ------"""

mu=X.mean()
sigma=X.std()
std_X=(X-mu)/sigma

std_X.fillna(0, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(std_X, y, random_state=1, train_size = 0.8)

"""------ Standardize Test Set ------"""
X_full_test = pd.read_csv("Clean Test.csv")
X_full_test.drop('Unnamed: 0',  axis = 1, inplace = True)

# Standardize the test set predictors
mu_test = X_full_test.mean()

sigma_test = X_full_test.std()

std_X_full = (X_full_test-mu_test)/sigma_test

std_X_full.fillna(0, inplace=True)

"""------ Huber Loss Regressor ------"""
from sklearn.linear_model import HuberRegressor

hbr = HuberRegressor()

hbr.fit(X_train, np.ravel(y_train))
print("HBR Fitting finished")

hbr_pred = hbr.predict(X_test)

hbr_rmse = np.sqrt(mean_squared_error(y_test, hbr_pred))

print("HBR RMSE = {0:.3f}".format(hbr_rmse))

"""------ TheilSen Regressor ------"""
tic = time.time()

from sklearn.linear_model import TheilSenRegressor

tsr = TheilSenRegressor()

tsr.fit(X_train, np.ravel(y_train))

toc = time.time()

print("TSR Training time: {0:.3f}".format(toc - tic))

tsr_pred = tsr.predict(X_test)

tsr_rmse = np.sqrt(mean_squared_error(y_test, tsr_pred))

print("TSR RMSE = {0:.3f}".format(tsr_rmse))

"""------ RANSAC Regressor ------"""
tic = time.time()

from sklearn.linear_model import RANSACRegressor

ransac = RANSACRegressor()

ransac.fit(X_train, np.ravel(y_train))

toc = time.time()

print("RANSAC Training time: {0:.3f}".format(toc - tic))

ran_pred = ransac.predict(X_test)

ran_rmse = np.sqrt(mean_squared_error(y_test, ran_pred))

print("RANSAC RMSE = {0:.3f}".format(ran_rmse))

"""------ Cross Validation ------"""
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

tic = time.time()
# Cross validator
kf=KFold(500, shuffle=True, random_state=1) 

# Methods
columns = ['CV RMSE']
rows = ['Huber']

regressions = [hbr]
results = pd.DataFrame(0.0, columns=columns, index=rows) # initialising a dataframe to hold the results

methods = {k: v for k, v in zip(rows, regressions)}

# Computing the results (Scikit Learn objects)
keys = ['Huber']
for key in keys:
    scores = cross_val_score(methods[key], X_train, np.ravel(y_train), cv=kf, 
                             scoring = 'neg_mean_squared_error')
    results.loc[key] = np.sqrt(-1*np.mean(scores))

toc = time.time()

print("Training time: {0:.3f}s".format(toc - tic))
print(results.round(3))

