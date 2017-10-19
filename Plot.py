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

X = pd.read_csv('Standardized X.csv')
y = pd.read_csv('Training response.csv')
std_X_full = pd.read_csv('Standardized test set.csv')

df = pd.concat([X, y], axis =1)

correlation = df.corr().SalePrice.sort_values(ascending = False)

# Pick the predictors with absolute correlation larger than 0.500
abs_correlation = np.absolute(correlation).sort_values(ascending = False)
large_correlation = abs_correlation[abs_correlation[:]>0.4]
small_correlation = abs_correlation[abs_correlation[:]<0.1]

labels = large_correlation.reset_index()
labels = labels[['index']]
labels.drop(0, axis =0, inplace=True)

i = 1
for a in labels['index']:
    plt.plot(X[a], y,'bo')
    plt.title(a)
#    plt.xlabel('MSSubClass')
    plt.ylabel('SalePrice')
    plt.grid(True)
    plt.show()
    i=i+1





##%%
#small_correlation = abs_correlation[abs_correlation[:]<0.09]
#
#"""------ Check correlation among 14 predictors ------"""
##predictors_14 = X[['GrLivArea', 'GarageArea', 'GarageCars','TotalBsmtSF', '1stFlrSF',
##                  'ExterQual_TA', 'YearBuilt', 'Foundation_PConc', 'OQ_8', 'ExterQual_Gd',
##                  'KitchenQual_TA', 'FullBath', 'YearRemod/Add', 'BsmtQual_Ex']]
##
##inter_correl = predictors_14.corr().round(3)
#
#"""------ Delete low correlation predictors ------"""
#lst = small_correlation.reset_index()
#del_predictors = lst[['index']]
#del_predictors.rename(columns = {'index': 'predictors'}, 
#                      inplace=True)
#del_predictors = del_predictors[del_predictors.predictors != 'Unnamed: 0']
#
#X_del = X
#
#for item in del_predictors['predictors']:
#    X_del.drop(item, axis = 1, inplace=True)
#    
#X = train.drop(['SalePrice', 'Unnamed: 0'], axis = 1) 
     
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
