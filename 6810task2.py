#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 14:02:40 2017

@author: xinyu
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 18:02:10 2017

@author: Administrator
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

df = pd.read_csv("Clothing_Store.csv") # load CSV file


### Data cleaning ###
X = df.iloc[:, :-1]


phone = pd.get_dummies(X['VALPHON'], drop_first=True) # Get dummy
phone = phone.rename(columns = {'Y':'VALPHON'})

X = X.drop('VALPHON', axis=1)
X = X.drop('HHKEY', axis=1)

# Cleaned Variables & target
X = pd.concat([X, phone], axis=1)
y = df.iloc[:,-1]

# Split the data into training and test sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state = 460475789)

X_train.to_csv('Clean X_train.csv')
y_train.to_csv('Clean y_train.csv')

X_test.to_csv('Clean X_test.csv')
y_test.to_csv('Clean y_test.csv')


### Data understanding ###
import scipy as sp
print(X_train.shape)

sp.sum(sp.isnan(y)) # missing value check

# Correlation matrix ##
corr_matrix = X_train.corr()
abs_corr_matrix = corr_matrix.abs()
large_corr = abs_corr_matrix >0.8
large_corr_t1 = abs_corr_matrix['STYLES']
large_corr_t2 = abs_corr_matrix['PROMOS']
large_corr_t3 = abs_corr_matrix['RESPONDED']  
large_corr_t4 = abs_corr_matrix['MON']
large_corr_t = pd.concat([large_corr_t1 , large_corr_t2 ,large_corr_t3, large_corr_t4], axis=1)      
        
large_corr_t.to_csv('multi_coli.csv')

# For Y:
# +ve %
print("Train set(% of +ve response): {0:.3f}%".format(float(sum(y_train))/float(len(y_train))*100))

# corr
corr_y = df.corr().RESP.sort_values(ascending = False)
corr_y_abs = df.corr().RESP.abs().sort_values(ascending = False)

corr_y.to_csv('corr to Response.csv')
corr_y.to_csv('corr to Response(absolute).csv')


# For X：
# Descriptive analysis + skew & kurt
descrip = X_train.describe()
descrip.loc['skew', :] = X_train.skew()
descrip.loc['kurt', :] = X_train.kurt()
print(descrip.round(3))

descrip.to_csv('descriptive for X_train.csv')

top5_corr=['FRE', 'CLASSES', 'STYLES', 'RESPONDED', 'RESPONSERATE']
top5_corr_descrip=descrip[['FRE', 'CLASSES', 'STYLES', 'RESPONDED', 'RESPONSERATE']]
top5_corr_descrip.to_csv('top5_corr_descrip.csv')

def hist(series):
    fig,ax=plt.subplots(figsize=(10,10))
    sns.distplot(series,ax=ax,hist_kws={'alpha':0.9, 'edgecolor':'black','color':sns.color_palette('Blues', n_colors=2)[-1]},
                                        kde_kws={'color':'black','alpha':0.7})
    return fig, ax

for i in range(5):
    hist(X_train[top5_corr[i]])
    sns.despine()   
    plt.show()
    
for i in range(5):
    sns.regplot(X_train[top5_corr[i]], y_train, color=sns.color_palette('Blues', n_colors=2)[-1],ci=None, logistic=True, y_jitter=0.05,
            scatter_kws={'s':25,'color':sns.color_palette('Blues')[-1],'alpha':0.5})                  
    sns.despine()    
    plt.show()



    
#data transformation
mu=X_train.mean()
sigma=X_train.std()

X_train=(X_train-mu)/sigma
X_test=(X_test-mu)/sigma

X_train.to_csv('Norm X_train.csv')
X_test.to_csv('Norm X_test.csv')




### Variable selection & Model prediction ### 
# http://songhuiming.github.io/pages/2016/07/12/variable-selection-in-python/

# =============================================================================
# Logistic Regression 
# http://blog.yhat.com/posts/logistic-regression-and-python.html    
import statsmodels.api as sm
logit = sm.Logit(y_train, sm.add_constant(X_train)).fit()
logit_sum = logit.summary()
print(logit_sum)
# From the summary, there are 15 varables had been selected. Df Model: 15. R^2 = 0.2668
# After normalization, 49 varable had been sleced. R^2 = 0.2677   ?????????

# Sort coefficent 
logit_params = logit.params.sort_values(ascending = False)
logit_params_abs = logit.params.abs().sort_values(ascending = False)

logit_params.to_csv('logit_coef.csv')
logit_params_abs.to_csv('logit_coef_asb.csv')

# Logistic regression
from sklearn.linear_model import LogisticRegression
logit = LogisticRegression()
logit.fit(X_train, y_train)

#predict
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, precision_score

logit_pred = logit.predict(X_test)
logit_prob = logit.predict_proba(X_test)


#print(classification_report(y_test, logit_pred))

logit_confusion = confusion_matrix(y_test, logit_pred)
#print(logit_confusion)
logit_mis = (1- accuracy_score(y_test, logit_pred)).round(3)
#print("Mis_rate_logit: {0:.3f}".format(logit_mis))

logit_se = np.sqrt(logit_mis*(1- logit_mis)/len(y_test)).round(3)
logit_sensi = logit_confusion[1,1]/np.sum(logit_confusion[1,:]).round(3)
logit_speci = logit_confusion[0,0]/np.sum(logit_confusion[0,:]).round(3)
logit_auc = roc_auc_score(y_test, logit_prob[:,1]).round(3)
logit_precision = precision_score(y_test, logit_pred).round(3)

columns = ['Logistic']
rows = ['Error rate', 'SE', 'Sensitivity', 'Specificity', 'AUC', 'Precision']

logit_score = pd.DataFrame([logit_mis, logit_se, logit_sensi, logit_speci, logit_auc, logit_precision], columns = columns, index = rows)


# =============================================================================

## Logistic Regression CV 'l1'
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
#For small datasets, ‘liblinear’ is a good choice. 'l1' =absolution loss function
#http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegressionCV.html
logit_l1 = LogisticRegressionCV(penalty = 'l1', solver = 'liblinear')
logit_l1.fit(X_train, y_train)  # fit training set


logit_l1_coef = pd.DataFrame(logit_l1.coef_)
logit_l1_coef_path = logit_l1.coefs_paths_
logit_l1_iter= logit_l1.n_iter_
logit_l1_scores = logit_l1.scores_

logit_l1_coef.to_csv('logitCV_coef')

# Predict
logit_l1_pred = logit_l1.predict(X_test)   # predit on test set
logit_l1_prob = logit_l1.predict_proba(X_test)


#print(classification_report(y_test, logit_l1_pred))
#print(confusion_matrix(y_test, logit_l1_pred))
#logit_l1_mis = 1- accuracy_score(y_test, logit_l1_pred)
#print("Mis_rate_logit: {0:.3f}".format(logit_l1_mis))
##logit_mse = mean_squared_error(y_test, logit_pred)  #compute R-square score & MSE
##print("logit RMSE = {0:.3f}".format(np.sqrt(logit_mse)))


logit_l1_confusion = confusion_matrix(y_test, logit_l1_pred)
logit_l1_mis = (1- accuracy_score(y_test, logit_l1_pred)).round(3)
logit_l1_se = np.sqrt(logit_l1_mis*(1- logit_l1_mis)/len(y_test)).round(3)
logit_l1_sensi = logit_l1_confusion[1,1]/np.sum(logit_l1_confusion[1,:]).round(3)
logit_l1_speci = logit_l1_confusion[0,0]/np.sum(logit_l1_confusion[0,:]).round(3)
logit_l1_auc = roc_auc_score(y_test, logit_l1_prob[:,1]).round(3)
logit_l1_precision = precision_score(y_test, logit_l1_pred).round(3)

columns = ['L1 regularised']
rows = ['Error rate', 'SE', 'Sensitivity', 'Specificity', 'AUC', 'Precision']

logit_l1_score = pd.DataFrame([logit_l1_mis, logit_l1_se, logit_l1_sensi, logit_l1_speci, logit_l1_auc, logit_l1_precision], columns = columns, index = rows)

# =============================================================================


# =============================================================================
# Decision Tree
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier 


# base on max_depth
# Grid Search
pipeline = Pipeline([('clf',DecisionTreeClassifier(criterion='entropy'))])
parameters = {'clf__max_depth': np.arange(2, 50, 1)}
grid_search = GridSearchCV(pipeline, parameters, cv=5)
grid_search.fit(X_train, y_train)
print("Best max_depths: {0}".format(grid_search.best_params_))

# fit model
tree1 = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 4, class_weight = 'balanced', random_state = 0)
tree1.fit(X_train, y_train)

# variable selected
tree1_import= pd.DataFrame([X.columns.values, tree1.feature_importances_]).T.sort(1, ascending = False)
tree1_import.to_csv('tree_importency_depths.csv')

# Predict
tree1_pred = tree1.predict(X_test)
tree1_prob = tree1.predict_proba(X_test)

#
#print(classification_report(y_test, tree_pred))
#print(confusion_matrix(y_test, tree_pred))
#tree_mis = 1- accuracy_score(y_test, tree_pred)
#print("Mis_rate_tree_depths: {0:.3f}".format(tree_mis))
##tree_mse = mean_squared_error(y_test, tree_pred)  #compute R-square score & MSE
##print("logit RMSE = {0:.3f}".format(np.sqrt(tree_mse)))


tree1_confusion = confusion_matrix(y_test, tree1_pred)
tree1_mis = (1- accuracy_score(y_test, tree1_pred)).round(3)
tree1_se = np.sqrt(tree1_mis*(1- tree1_mis)/len(y_test)).round(3)
tree1_sensi = tree1_confusion[1,1]/np.sum(tree1_confusion[1,:]).round(3)
tree1_speci = tree1_confusion[0,0]/np.sum(tree1_confusion[0,:]).round(3)
tree1_auc = roc_auc_score(y_test, tree1_prob[:,1]).round(3)
tree1_precision = precision_score(y_test, tree1_pred).round(3)

columns = ['Decision Tree_depths']
rows = ['Error rate', 'SE', 'Sensitivity', 'Specificity', 'AUC', 'Precision']

tree1_score = pd.DataFrame([tree1_mis, tree1_se, tree1_sensi, tree1_speci, tree1_auc, tree1_precision], columns = columns, index = rows)


# =============================================================================

# based on max_features
# Grid Search
pipeline = Pipeline([('clf',DecisionTreeClassifier(criterion='entropy'))])
parameters = {'clf__max_features': np.arange(2, 50, 1)}
grid_search = GridSearchCV(pipeline, parameters, cv=5)
grid_search.fit(X_train, y_train)
print("Best max_features: {0}".format(grid_search.best_params_))

# fit model
tree2 = tree.DecisionTreeClassifier(criterion = 'entropy', max_features = 36, class_weight = 'balanced', random_state = 0)
tree2.fit(X_train, y_train)

# variable selected
tree2_import= pd.DataFrame([X.columns.values, tree2.feature_importances_]).T.sort(1, ascending = False)
tree2_import.to_csv('tree_importency_featues.csv')

# Predict
tree2_pred = tree2.predict(X_test)
tree2_prob = tree2.predict_proba(X_test)

#print(classification_report(y_test, tree_pred))
#print(confusion_matrix(y_test, tree_pred))
#tree_mis = 1- accuracy_score(y_test, tree_pred)
#print("Mis_rate_tree_features: {0:.3f}".format(tree_mis))

# 0.291 vs. 0.203 Mis rates. Featues has less mis rate. 

tree2_confusion = confusion_matrix(y_test, tree2_pred)
tree2_mis = (1- accuracy_score(y_test, tree2_pred)).round(3)
tree2_se = np.sqrt(tree2_mis*(1- tree2_mis)/len(y_test)).round(3)
tree2_sensi = tree2_confusion[1,1]/np.sum(tree2_confusion[1,:]).round(3)
tree2_speci = tree2_confusion[0,0]/np.sum(tree2_confusion[0,:]).round(3)
tree2_auc = roc_auc_score(y_test, tree2_prob[:,1]).round(3)
tree2_precision = precision_score(y_test, tree2_pred).round(3)

columns = ['Decision Tree_features']
rows = ['Error rate', 'SE', 'Sensitivity', 'Specificity', 'AUC', 'Precision']

tree2_score = pd.DataFrame([tree2_mis, tree2_se, tree2_sensi, tree2_speci, tree2_auc, tree2_precision], columns = columns, index = rows)


# =============================================================================





#[先不要了]
## Recursive feature elimination with cross-validation
## http://scikit-learn.org/stable/modules/feature_selection.html#rfe
#from sklearn.svm import SVC
#from sklearn.model_selection import StratifiedKFold
#from sklearn.feature_selection import RFECV
## Create the RFE object and compute a cross-validated score.
#svc = SVC(kernel="linear")
## The "accuracy" scoring is proportional to the number of correct
## classifications
#rfecv = RFECV(estimator=svc, step=1, cv=5, scoring='accuracy')
#rfecv.fit(X_train, y_train)
#
#print("Optimal number of features : %d" % rfecv.n_features_)


# =============================================================================
## Forward selection
#from statlearning import forward  # Forward method
#
## Fit model
#fwd = forward()
#fwd.fit(X_train, y_train)
#
## Feature select
#idx = fwd.subset
#print(idx)
#X_train[idx].head(5)
## 20 features has been slected.
#
## predict
#fwd_pred = fwd.predict(X_test)
#
##ERROR!!! # Mix type of y not allowed, got types {'binary', 'continuous'}
#
##print(classification_report(y_test, fwd_pred))
##print(confusion_matrix(y_test, fwd_pred))
##fwd_mis = 1- accuracy_score(y_test, fwd_pred)
##print("Mis_rate_fwd: {0:.3f}".format(fwd_mis))
# =============================================================================


# PCA

# https://www.analyticsvidhya.com/blog/2016/03/practical-guide-principal-component-analysis-python/
# =============================================================================
# from sklearn.decomposition import PCA
# pca = PCA(n_components = 20)
# pca.fit(X_train, y_train)
# 
# 
# var=pca.explained_variance_ratio_
# print(var)
# print(pca.components_)
# aaa = pca.components_
# aaa= pd.DataFrame(aaa)
# aaa= aaa.abs()
# a = aaa[0].sort_values(ascending = False)
# 
# 
# var1=np.cumsum(pca.explained_variance_ratio_)
# print(var1)
# 
# k = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
# plt.plot(var1)
# plt.xticks(k)
# plt.show()
## 
# k = np.arange(1, 20)
# 
# # =============================================================================

#
#



 
# =============================================================================
# Knn
from sklearn.model_selection import GridSearchCV
from sklearn import neighbors

#Gridsearch, model fit
k_grid = {"n_neighbors": np.arange(1, 32, 1)}
knn = GridSearchCV(neighbors.KNeighborsClassifier(), k_grid, cv = 5)
knn.fit(X_train, y_train)
knn_cv = knn.best_estimator_
print(knn_cv)

# predict
knn_pred = knn.predict(X_test)
knn_prob = knn.predict_proba(X_test)

#print(classification_report(y_test, knn_pred))
#print(confusion_matrix(y_test, knn_pred))
#knn_mis = 1- accuracy_score(y_test, knn_pred)
#print("Mis_rate_fwd: {0:.3f}".format(knn_mis))
## mis_rate: 0.162

knn_confusion = confusion_matrix(y_test, knn_pred)
knn_mis = (1- accuracy_score(y_test, knn_pred)).round(3)
knn_se = np.sqrt(knn_mis*(1- knn_mis)/len(y_test)).round(3)
knn_sensi = knn_confusion[1,1]/np.sum(knn_confusion[1,:]).round(3)
knn_speci = knn_confusion[0,0]/np.sum(knn_confusion[0,:]).round(3)
knn_auc = roc_auc_score(y_test, knn_prob[:,1]).round(3)
knn_precision = precision_score(y_test, knn_pred).round(3)

columns = ['KNN']
rows = ['Error rate', 'SE', 'Sensitivity', 'Specificity', 'AUC', 'Precision']

knn_score = pd.DataFrame([knn_mis, knn_se, knn_sensi, knn_speci, knn_auc, knn_precision], columns = columns, index = rows)
# =============================================================================


# =============================================================================
# Adaboost Classifier [跑不动]
from sklearn import ensemble

n_grid = {"n_estimators": np.arange(1, 50, 1)}
ada = GridSearchCV(ensemble.AdaBoostClassifier(algorithm='SAMME'), n_grid, cv = 5)
ada.fit(X_train, y_train)
ada_cv = ada.best_estimator_
print(ada_cv)

# predict
ada_pred = ada.predict(X_test)
ada_prob = ada.predict_proba(X_test)

#print(classification_report(y_test, ada_pred))
#print(confusion_matrix(y_test, ada_pred))
#ada_mis = 1- accuracy_score(y_test, ada_pred)
#print("Mis_rate_fwd: {0:.3f}".format(ada_mis))

ada_confusion = confusion_matrix(y_test, ada_pred)
ada_mis = (1- accuracy_score(y_test, ada_pred)).round(3)
ada_se = np.sqrt(ada_mis*(1- ada_mis)/len(y_test)).round(3)
ada_sensi = ada_confusion[1,1]/np.sum(ada_confusion[1,:]).round(3)
ada_speci = ada_confusion[0,0]/np.sum(ada_confusion[0,:]).round(3)
ada_auc = roc_auc_score(y_test, ada_prob[:,1]).round(3)
ada_precision = precision_score(y_test, ada_pred).round(3)

columns = ['AdaBoost']
rows = ['Error rate', 'SE', 'Sensitivity', 'Specificity', 'AUC', 'Precision']

ada_score = pd.DataFrame([ada_mis, ada_se, ada_sensi, ada_speci, ada_auc, ada_precision], columns = columns, index = rows)

# =============================================================================


# =============================================================================
### ~N  Dscriminant analysis
X_train_da = X_train.drop(['CC_CARD','VALPHON','WEB'], axis=1)
X_test_da = X_test.drop(['CC_CARD','VALPHON','WEB'], axis=1)

# LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit(X_train_da, y_train)

# predict
lda_pred = lda.predict(X_test_da)
lda_prob = lda.predict_proba(X_test_da)

#print(classification_report(y_test, lda_pred))
#print(confusion_matrix(y_test, lda_pred))
#lda_mis = 1- accuracy_score(y_test, lda_pred)
#print("Mis_rate_lda: {0:.3f}".format(lda_mis))
#
# 0.155 miss

lda_confusion = confusion_matrix(y_test, lda_pred)
lda_mis = (1- accuracy_score(y_test, lda_pred)).round(3)
lda_se = np.sqrt(lda_mis*(1- lda_mis)/len(y_test)).round(3)
lda_sensi = lda_confusion[1,1]/np.sum(lda_confusion[1,:]).round(3)
lda_speci = lda_confusion[0,0]/np.sum(lda_confusion[0,:]).round(3)
lda_auc = roc_auc_score(y_test, lda_prob[:,1]).round(3)
lda_precision = precision_score(y_test, lda_pred).round(3)

columns = ['LDA']
rows = ['Error rate', 'SE', 'Sensitivity', 'Specificity', 'AUC', 'Precision']

lda_score = pd.DataFrame([lda_mis, lda_se, lda_sensi, lda_speci, lda_auc, lda_precision], columns = columns, index = rows)



# =============================================================================
# QDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train_da, y_train)

# predict
qda_pred = qda.predict(X_test_da)
qda_prob = qda.predict_proba(X_test_da)

#print(classification_report(y_test, qda_pred))
#print(confusion_matrix(y_test, qda_pred))
#qda_mis = 1- accuracy_score(y_test, qda_pred)
#print("Mis_rate_qda: {0:.3f}".format(qda_mis))
# 0.201 miss


qda_confusion = confusion_matrix(y_test, qda_pred)
qda_mis = (1- accuracy_score(y_test, qda_pred)).round(3)
qda_se = np.sqrt(qda_mis*(1- qda_mis)/len(y_test)).round(3)
qda_sensi = qda_confusion[1,1]/np.sum(qda_confusion[1,:]).round(3)
qda_speci = qda_confusion[0,0]/np.sum(qda_confusion[0,:]).round(3)
qda_auc = roc_auc_score(y_test, qda_prob[:,1]).round(3)
qda_precision = precision_score(y_test, qda_pred).round(3)

columns = ['QDA']
rows = ['Error rate', 'SE', 'Sensitivity', 'Specificity', 'AUC', 'Precision']

qda_score = pd.DataFrame([qda_mis, qda_se, qda_sensi, qda_speci, qda_auc, qda_precision], columns = columns, index = rows)


# =============================================================================
# regulization
alphas = np.linspace(0, 1, 21)
param_grid = {'reg_param': alphas}  
gscv = GridSearchCV(QuadraticDiscriminantAnalysis(), param_grid, cv = 5, scoring = 'accuracy')
gscv.fit(X_train_da, y_train)

qda_reg = gscv.best_estimator_

# predict
qda_reg_pred = qda_reg.predict(X_test_da)
qda_reg_prob = qda_reg.predict_proba(X_test_da)

#print(classification_report(y_test, qda_reg_pred))
#print(confusion_matrix(y_test, qda_reg_pred))
#qda_reg_mis = 1- accuracy_score(y_test, qda_reg_pred)
#print("Mis_rate_qda_reg: {0:.3f}".format(qda_reg_mis))
# 0.191 miss

qda_reg_confusion = confusion_matrix(y_test, qda_reg_pred)
qda_reg_mis = (1- accuracy_score(y_test, qda_reg_pred)).round(3)
qda_reg_se = np.sqrt(qda_reg_mis*(1- qda_reg_mis)/len(y_test)).round(3)
qda_reg_sensi = qda_reg_confusion[1,1]/np.sum(qda_reg_confusion[1,:]).round(3)
qda_reg_speci = qda_reg_confusion[0,0]/np.sum(qda_reg_confusion[0,:]).round(3)
qda_reg_auc = roc_auc_score(y_test, qda_reg_prob[:,1]).round(3)
qda_reg_precision = precision_score(y_test, qda_reg_pred).round(3)

columns = ['Regularised QDA']
rows = ['Error rate', 'SE', 'Sensitivity', 'Specificity', 'AUC', 'Precision']

qda_reg_score = pd.DataFrame([qda_reg_mis, qda_reg_se, qda_reg_sensi, qda_reg_speci, qda_reg_auc, qda_reg_precision], columns = columns, index = rows)



# =============================================================================


# =============================================================================
### Emsamble(w/o Dscriminant analysis) [跑不动]
from sklearn.ensemble import VotingClassifier
# Voting Classifier
eclf = VotingClassifier(estimators = [('Logit', logit), ('Logit_L1', logit_l1), ('Tree', tree), ('kNN', knn), ('Adabost', ada)], voting = 'hard')

eclf.fit(X_train, y_train)

#predict
eclf_pred= eclf.predict(X_test)
eclf_prob = eclf.predict_proba(X_test)

#print(classification_report(y_test, eclf_pred))
#print(confusion_matrix(y_test, eclf_pred))
#eclf_mis = 1- accuracy_score(y_test, eclf_pred)
#print("Mis_rate_eclf: {0:.3f}".format(eclf_mis))

eclf_confusion = confusion_matrix(y_test, eclf_pred)
eclf_mis = (1- accuracy_score(y_test, eclf_pred)).round(3)
eclf_se = np.sqrt(eclf_mis*(1- eclf_mis)/len(y_test)).round(3)
eclf_sensi = eclf_confusion[1,1]/np.sum(eclf_confusion[1,:]).round(3)
eclf_speci = eclf_confusion[0,0]/np.sum(eclf_confusion[0,:]).round(3)
eclf_auc = roc_auc_score(y_test, eclf_prob[:,1]).round(3)
eclf_precision = precision_score(y_test, eclf_pred).round(3)

columns = ['Emsamble']
rows = ['Error rate', 'SE', 'Sensitivity', 'Specificity', 'AUC', 'Precision']

eclf_score = pd.DataFrame([eclf_mis, eclf_se, eclf_sensi, eclf_speci, eclf_auc, eclf_precision], columns = columns, index = rows)

# =============================================================================

# =============================================================================
# Neural network models（Multi-layer Perceptron）
# http://scikit-learn.org/stable/modules/neural_networks_supervised.html
from sklearn.neural_network import MLPClassifier

nn = MLPClassifier(activation = 'logistic')

nn.fit(X_train, y_train) 

#predict
nn_pred = nn.predict(X_test)
nn_prob = nn.predict_proba(X_test)

nn_confusion = confusion_matrix(y_test, nn_pred)
nn_mis = (1- accuracy_score(y_test, nn_pred)).round(3)
nn_se = np.sqrt(nn_mis*(1- nn_mis)/len(y_test)).round(3)
nn_sensi = nn_confusion[1,1]/np.sum(nn_confusion[1,:]).round(3)
nn_speci = nn_confusion[0,0]/np.sum(nn_confusion[0,:]).round(3)
nn_auc = roc_auc_score(y_test, nn_prob[:,1]).round(3)
nn_precision = precision_score(y_test, nn_pred).round(3)

columns = ['Emsamble']
rows = ['Error rate', 'SE', 'Sensitivity', 'Specificity', 'AUC', 'Precision']

nn_score = pd.DataFrame([nn_mis, nn_se, nn_sensi, nn_speci, nn_auc, nn_precision], columns = columns, index = rows)

# =============================================================================



# =============================================================================
### Model evaluation[没有ada yet]
summary = pd.concat([logit_score, logit_l1_score, tree1_score, tree2_score, knn_score, nn_score, lda_score, qda_score, qda_reg_score], axis=1)                

summary.to_csv('Model evaluation_summary.csv')


### Confidence Interval!!!


### ROC curves   [ada & knn not included yet]
# https://xkcd.com/color/rgb/    选颜色
palette = ['#1F77B4', '#FF7F0E', '#2CA02C', '#DB2728', '#9467BD', '#95A5A6', '#34495E', 'denim blue', 'pale red',]

from sklearn.metrics import roc_curve

labels=['Logistic regression', 'L1 regularised', 'Tree_depths', 'Tree_features', 'Neural network', 'LDA', 'QDA', 'Regularised QDA']
methods=[logit, logit_l1, tree1, tree2, nn, lda, qda, qda_reg]

fig, ax= plt.subplots(figsize=(9,6))

for i, method in enumerate(methods):
    if i < 5:
        y_prob = method.predict_proba(X_test)
    else:
        y_prob = method.predict_proba(X_test_da)
        
    fpr, tpr, _ = roc_curve(y_test, y_prob[:,1])
    auc = roc_auc_score(y_test, y_prob[:,1])
    ax.plot(1-fpr, tpr, label=labels[i] + ' (AUC = {:.3f})'.format(auc), color = palette[i])

    
ax.set_xlabel('Specificity')
ax.set_ylabel('Sensitivity')
ax.set_title('ROC curves', fontsize=14)
plt.legend(fontsize=13)
plt.show()





