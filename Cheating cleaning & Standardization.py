# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 20:35:49 2017

@author: Grace
"""

#%%
import pandas as pd
import numpy as np

train = pd.read_csv('Train set from Kaggle.csv').drop('SalePrice', axis=1)
test = pd.read_csv('Test set from Kaggle.csv').drop('Id', axis=1)

#train_dum = pd.get_dummies(train)
#test_dum = pd.get_dummies(test)
#
df = pd.concat([train, test])

df = pd.get_dummies(df)

MS_subclass = pd.get_dummies(df.iloc[:,0], drop_first = True)

MS_subclass.rename(columns={20:'MS_020', 30:'MS_030', 40:'MS_040', 45:'MS_045', 
                            50:'MS_050', 60:'MS_060', 70:'MS_070', 75:'MS_075', 
                            80:'MS_080', 85:'MS_085', 90:'MS_090', 120:'MS_120',
                            150:'MS_150', 160:'MS_160', 180:'MS_180', 190:'MS_190'},inplace=True)

Overall_Q = pd.get_dummies(df.iloc[:,3], drop_first = True)

Overall_Q.rename(columns={1:'OQ_1',2: 'OQ_2', 3: 'OQ_3', 4:'OQ_4', 5:'OQ_5', 6:'OQ_6',
                               7:'OQ_7', 8:'OQ_8', 9:'OQ_9', 10:'OQ_10'}, inplace = True)

Overall_C = pd.get_dummies(df.iloc[:, 4], drop_first = True)

Overall_C.rename(columns={1:'OC_1',2: 'OC_2', 3: 'OC_3', 4:'OC_4', 5:'OC_5', 6:'OC_6',
                               7:'OC_7', 8:'OC_8', 9:'OC_9', 10:'OC_10'}, inplace = True)

df.drop(['MSSubClass', 'OverallQual', 'OverallCond'], inplace=True, axis = 1)

df_clean = pd.concat([df, MS_subclass, Overall_Q, Overall_C], axis = 1)

train_set = df_clean.iloc[:804, :]
test_set = df_clean.iloc[804:, :]

saleprice = pd.read_csv('Train set from Kaggle.csv').iloc[:, -1]
train_set = pd.concat([train_set, saleprice], axis = 1)
train_set.fillna(0, inplace=True)

test_set.fillna(0, inplace = True)

train_set.to_csv('Clean Train.csv')
test_set.to_csv('Clean Test.csv')

#%%
"""------- Standardize the training set ------"""
y = pd.DataFrame(train_set.iloc[:, -1])
X = train_set.drop(['SalePrice'], axis = 1) 

mu=X.mean()
sigma=X.std()
std_X=(X-mu)/sigma
std_X.fillna(0, inplace=True)
std_X.to_csv('Standardized X.csv')

# Save training y
y.to_csv('Training response.csv')

"""------ Standardize the test set ------"""
mu_test = test_set.mean()

sigma_test = test_set.std()

std_X_full = (test_set - mu_test)/sigma_test

std_X_full.fillna(0, inplace=True)
std_X_full.to_csv('Standardized test set.csv')
