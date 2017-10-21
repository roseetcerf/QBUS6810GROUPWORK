# -*- coding: utf-8 -*-
"""
Created on Sat Oct 21 18:21:35 2017

@author: Grace
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

X = pd.read_csv('Clean Train.csv').iloc[:,1:-1]
y = pd.read_csv('Training response.csv')

correl = pd.concat([X, y], axis = 1).corr().SalePrice.sort_values(ascending = False).drop('SalePrice')
abs_correl = abs(correl).sort_values(ascending = False).reset_index()
correl = correl.reset_index()
large_correl = abs_correl.head(10)

abs_correl = abs_correl.fillna(0)
nozero_cor = abs_correl.loc[abs_correl['SalePrice']!=0]
small_correl = nozero_cor.tail(10)

true = pd.DataFrame()
for a in large_correl['index']:
    k = pd.DataFrame(correl.loc[(correl['index']==a), 'SalePrice'])
    true = true.append(k)
true.columns = ['Correlation']
true = true.reset_index().drop('index', axis =1)
large_correl.loc[:,'Correl'] = true.loc[:,'Correlation']

true = pd.DataFrame()
for a in small_correl['index']:
    k = pd.DataFrame(correl.loc[(correl['index']==a), 'SalePrice'])
    true = true.append(k)
true.columns = ['Correlation']
true = true.reset_index().drop('index', axis =1)
small_correl = small_correl.reset_index()
small_correl.loc[:,'Correl'] = true.loc[:,'Correlation']
small_correl = small_correl.drop('level_0', axis=1)

large_sub = pd.DataFrame()
for a in large_correl['index']:
    large_sub=large_sub.append(X.loc[:,a])
large_sub = large_sub.T

small_sub = pd.DataFrame()
for a in small_correl['index']:
    small_sub = small_sub.append(X.loc[:,a])
small_sub = small_sub.T

describe_lar = large_sub.describe()
describe_lar.loc['skew',:] = large_sub.skew()
describe_lar.loc['kurt',:] = large_sub.kurt()

describe_sml = small_sub.describe()
describe_sml.loc['skew',:] = small_sub.skew()
describe_sml.loc['kurt',:] = small_sub.kurt()

df = pd.concat([X,y], axis=1)
for a in small_correl['index']:
    fig, ax = plt.subplots(figsize=(9,6))
    sns.regplot(x=a, y='SalePrice', data=df,
            scatter_kws = {'s': 25}, lowess=True, color='cornflowerblue')
    sns.despine()
    fig.savefig('Low correlation scatter& reg {0}.pdf'.format(a))

train = pd.read_csv('Train set from Kaggle.csv')

Ext_Q = train.groupby('ExterQual')['SalePrice'].describe()
Ext_Q_des = pd.DataFrame(Ext_Q)

fig, ax = plt.subplots(figsize=(9,6))
sns.boxplot(x='ExterQual', y='SalePrice', data=train, palette='Blues')
plt.tight_layout()
plt.show()
fig.savefig('Boxplot of ExterQual.pdf')

Over_C = train.groupby('OverallCond')['SalePrice'].describe()
Over_C = pd.DataFrame(Over_C)
Over_C_re = np.reshape(np.array(Over_C), (7,8))

fig, ax = plt.subplots(figsize=(9,6))
sns.boxplot(x='OverallCond', y='SalePrice', data=train, palette='Blues')
plt.tight_layout()
plt.show()
fig.savefig('Boxplot of OverallCond.pdf')



