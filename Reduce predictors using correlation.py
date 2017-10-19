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

train = pd.read_csv('Clean Train.csv')
y = pd.DataFrame(train.iloc[:, -1])
X = train.drop(['SalePrice', 'Unnamed: 0'], axis = 1) 

std_X_full = pd.read_csv('Standardized test set.csv')


#%%
"""------ Correlation with response ------"""
descriptive_y = y.describe()
descriptive_y.loc["skew", :] = y.skew()
descriptive_y.loc["kurt", :] = y.kurt()

correlation = train.corr().SalePrice.sort_values(ascending = False)

# Pick the predictors with absolute correlation larger than 0.500
abs_correlation = np.absolute(correlation).sort_values(ascending = False)
large_correlation = abs_correlation[abs_correlation[:]>0.5]

small_correlation = abs_correlation[abs_correlation[:]<0.02]

#%%
"""------ Check correlation among 14 predictors ------"""
#predictors_14 = X[['GrLivArea', 'GarageArea', 'GarageCars','TotalBsmtSF', '1stFlrSF',
#                  'ExterQual_TA', 'YearBuilt', 'Foundation_PConc', 'OQ_8', 'ExterQual_Gd',
#                  'KitchenQual_TA', 'FullBath', 'YearRemod/Add', 'BsmtQual_Ex']]
#
#inter_correl = predictors_14.corr().round(3)

"""------ Delete low correlation predictors ------"""
lst = small_correlation.reset_index()
del_predictors = lst[['index']]
del_predictors.rename(columns = {'index': 'predictors'}, 
                      inplace=True)
del_predictors = del_predictors[del_predictors.predictors != 'Unnamed: 0']

X_del = X

for item in del_predictors['predictors']:
    X_del.drop(item, axis = 1, inplace=True)
    
X = train.drop(['SalePrice', 'Unnamed: 0'], axis = 1) 
     
#%%
"""------ Fit LASSO CV ------"""
mu=X_del.mean()
sigma=X_del.std()
std_X=(X_del-mu)/sigma
std_X.fillna(0, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(std_X, y, random_state=1, train_size = 0.8)

lasso = LassoCV(cv = 100)

lasso.fit(X_train, np.ravel(y_train))

lasso_pred = lasso.predict(X_test)

lasso_mse = mean_squared_error(y_test, lasso_pred)

print("LASSO RMSE = {0:.3f}".format(np.sqrt(lasso_mse)))

columns = X.columns.values
lasso_cols = pd.Series(columns[np.nonzero(lasso.coef_)])

"""------ Refit full train set and predict ------"""
for item in del_predictors['predictors']:
    std_X_full.drop(item, axis = 1, inplace=True)

lasso.fit(std_X, np.ravel(y))

lasso_pred_full = lasso.predict(std_X_full)
pd.Series(lasso_pred_full).to_csv('Submission 11.csv')

columns = X.columns.values
lasso_cols = pd.Series(columns[np.nonzero(lasso.coef_)])



 #%%                      
#""" Distribution plot """
#plt.figure(figsize = (20,15))
#plt.hist(np.ravel(y), 20, facecolor='cornflowerblue', edgecolor = 'k', alpha=0.75)
#plt.xlabel('Sale Price', fontsize = 35)
#plt.ylabel('Number of Observations', fontsize = 35)
#plt.title('Histogram of Sale Price', fontsize = 35)
#plt.axis([0, 650000, 0, 250], fontsize = 30)
#plt.xticks(fontsize = 20)
#plt.yticks(fontsize = 20)
#plt.grid(True)
#plt.savefig('Histogram of Sale Price.pdf')
