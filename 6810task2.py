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

df = pd.read_csv("D:\Semester 3\QBUS 6810\Group Assignment\Task 2\QBUS6810GROUPWORK-task2/Clothing_Store.csv") # load CSV file


### Data cleaning ###
X = df.iloc[:, :-1]


phone = pd.get_dummies(X['VALPHON'], drop_first=True) # Get dummy
phone = phone.rename(columns = {'Y':'VALPHON'})

X = X.drop('VALPHON', axis=1)
#X = X.drop('HHKEY', axis=1)

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







# For Y:
# +ve %
print("Proportion of positive response in the training set: {0:.3f}%"
      .format(float(sum(y_train))/float(len(y_train))*100))

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
top5_corr_descrip.to_csv("D:\Semester 3\QBUS 6810\Group Assignment\Task 2\QBUS6810GROUPWORK-task2/top5_corr_descrip.csv")

def hist(series):
    fig,ax=plt.subplots()
    sns.distplot(series,ax=ax,hist_kws={'alpha':0.9, 'edgecolor':'black','color':sns.color_palette('Blues')[-1]},
                                        kde_kws={'color':'black','alpha':0.7})
    return fig, ax

for i in range(5):
    hist(X_train[top5_corr[i]])
    sns.despine()
    plt.show()
        
for i in range(5):
    sns.regplot(X_train[top5_corr[i]], y_train, color=sns.color_palette('Blues')[-1],ci=None, logistic=True, y_jitter=0.05,
            scatter_kws={'s':25,'color':sns.color_palette('Blues')[-1],'alpha':0.5})                  
    sns.despine()
    plt.show()
    
    
#data transformation
mu=X_train.mean()
sigma=X_train.std()

X_train=(X_train-mu)/sigma
X_test=(X_test-mu)/sigma


### Variable selection ### 
# http://songhuiming.github.io/pages/2016/07/12/variable-selection-in-python/

# Logistic Regression 
# http://blog.yhat.com/posts/logistic-regression-and-python.html

    
import statsmodels.api as sm
logit = sm.Logit(y_train, sm.add_constant(X_train)).fit()
logit_sum = logit.summary()
print(logit_sum)
# From the summary, there are 15 varables had been selected. Df Model: 15. R^2 = 0.2668

# Sort coefficent 
logit_params = logit.params.sort_values(ascending = False)
logit_params_abs = logit.params.abs().sort_values(ascending = False)

logit_params.to_csv('logit_coef.csv')
logit_params_abs.to_csv('logit_coef_asb.csv')



# Decision Tree
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier 
# Grid Search
pipeline = Pipeline([('clf',DecisionTreeClassifier(criterion='entropy'))])
parameters = {'clf__max_depth': np.arange(2, 50, 1)}

grid_search = GridSearchCV(pipeline, parameters, cv=5)
grid_search.fit(X_train, y_train)
print("Best score: {0:.3f}".format(grid_search.best_score_))
print("Best max_depth: {0}".format(grid_search.best_params_))

tree = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 4, class_weight = 'balanced', random_state = 0)

tree.fit(X_train, y_train)

tree_import= pd.DataFrame([X.columns.values, tree.feature_importances_]).T.sort(1, ascending = False)

tree_import.to_csv('tree_importency.csv')



# Recursive feature elimination with cross-validation
# http://scikit-learn.org/stable/modules/feature_selection.html#rfe
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
# Create the RFE object and compute a cross-validated score.
svc = SVC(kernel="linear")
# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv = RFECV(estimator=svc, step=1, cv=5, scoring='accuracy')
rfecv.fit(X_train, y_train)

print("Optimal number of features : %d" % rfecv.n_features_)



# Forward selection
from statlearning import forward  # Forward method

fwd = forward()
fwd.fit(X_train, y_train)

idx = fwd.subset
print(idx)

X_train[idx].head(5)


# Forward variable selection　　［还没看懂呢］
import statsmodels.formula.api as smf
xvar = X.columns.tolist()
yvar = 'cum_pd_num'   ####？？？？？？？
def importance_foreward(indata = df, yVar = yvar, xVar = xvar, stopn = 4):

    flist = []
    nx = min(len(xVar), stopn)

    while len(flist) < nx:
        best_score = -np.inf
        for i in xVar:
            newflist = flist + [i]
            f = yVar + ' ~ ' + '+'.join(newflist)
            reg = smf.ols(formula = str(f), data = indata).fit()
            score = reg.fvalue
            if score > best_score:
                best_score, record_i, record_newflist = score, i, newflist
        flist = record_newflist
        print (flist)
        xVar.remove(record_i)
        print (len(xVar))
    finmodel =  smf.ols(formula = str(yVar + ' ~ ' + '+'.join(flist)), data = indata).fit()
    print (finmodel.summary())
    return flist



# PCA

# https://www.analyticsvidhya.com/blog/2016/03/practical-guide-principal-component-analysis-python/
# =============================================================================
 from sklearn.decomposition import PCA
 pca = PCA(n_components = 20)
 pca.fit(X_train, y_train)
 
 
 var=pca.explained_variance_ratio_
 print(var)
 print(pca.components_)
 aaa = pca.components_
 aaa= pd.DataFrame(aaa)
 aaa= aaa.abs()
 a = aaa[0].sort_values(ascending = False)
 
 
 var1=np.cumsum(pca.explained_variance_ratio_)
 print(var1)
 
 k = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
 plt.plot(var1)
 plt.xticks(k)
 plt.show()
# 
 k = np.arange(1, 20)
 
 =============================================================================

#
#
# Logistic Regression Regulization   [跑不动！！！]
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import mean_squared_error

#For small datasets, ‘liblinear’ is a good choice. 'l1' =absolution loss function
logit_l1 = LogisticRegressionCV(penalty = 'l1', solver = 'liblinear')
logit_l1.fit(X_train, y_train)  # fit training set
logit_pred = logit_l1.predict(X_test)   # predit on test set
logit_mse = mean_squared_error(y_test, logit_pred)  #compute R-square score & MSE
print("logit RMSE = {0:.3f}".format(np.sqrt(logit_mse)))

columns = X.columns.values
aaa = logit_l1.coef_
aaa = np.nonzero(logit_l1.coef_)
logit_cols = pd.Series(columns[np.nonzero(logit_l1.coef_)])   
print('Logit features: ', logit_cols)                  


# Knn
from sklearn.model_selection import GridSearchCV
from sklearn import neighbors

k_grid = {"n_neighbors": np.arange(1, 32, 1)}
knn = GridSearchCV(neighbors.KNeighborsClassifier(), k_grid, cv = 5)
knn.fit(X_train, y_train)
knn_cv = knn.best_estimator_
print(knn_cv)

# KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           #metric_params=None, n_jobs=1, n_neighbors=24, p=2, weights='uniform')




from sklearn.tree import DecisionTreeRegressor
def importance_dt():
    tree = DecisionTreeRegressor()
    tree = tree.fit(X_train, y_train)
#    print mean_squared_error(tree.predict(X_train), Y_train)
#    print tree.score(X_train, y_train)
    importance_dt = pd.DataFrame([X.columns.values, tree.feature_importances_]).T.sort(1, ascending = False)
    return importance_dt



# Histograms
import seaborn as sns
def hist(series):
    fig, ax = plt.subplots()
    sns.distplot(series, ax = ax, hist_kws = {'alpha': 0.9, 'edgecolor': 'black', 'color': sns.color_palette('Blues') [-1]},
                                              ked_kws = {'color': 'black', 'alpha':0.7})
    return fig, ax



# 必须先弄多少个predictor说
plt.hist(X_train['POUTERWEAR'])




#  你可以自己跑几个看一下就知道了
sns.regplot(X_train['PBLOUSES'], y_train, color=sns.color_palette('Blues')[-1], ci=None, logistic=True, y_jitter=0.05, 
            scatter_kws={'s': 25, 'color': sns.color_palette('Blues')[-1], 'alpha': .5})
sns.despine()
plt.show()


# Predictor processing
                        


print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
logit_l1_mis = 1- accuracy_score(y_test, y_pred)
print(logit_l1_mis)



# KNN
from sklearn.model_selection import GridSearchCV
from sklearn import neighbors

k_grid = {"n_neighbors": np.arange(1, 32, 1)}
knn = GridSearchCV(neighbors.KNeighborsClassifier(), k_grid, cv = 5)
knn.fit(X_train, y_train)
knn_cv = knn.best_estimator_
print(knn_cv)

y_pred = knn.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
knn_mis = 1- accuracy_score(y_test, y_pred)
print(knn_mis)

# Decision Trees
from sklearn import tree

d_grid = {"max_depth": np.arange(2, 51, 1)}
tree = GridSearchCV(tree.DecisionTreeClassifier(), d_grid, cv = 5)
tree.fit(X_train, y_train)
tree_cv = tree.best_estimator_

y_pred = tree.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
tree_mis = 1- accuracy_score(y_test, y_pred)
print(tree_mis)


# Adaboost Classifier
from sklearn import ensemble

n_grid = {"n_estimators": np.arange(1, 101, 1)}
ada = GridSearchCV(ensemble.AdaBoostClassifier(algorithm='SAMME'), n_grid, cv = 5)
ada.fit(X_train, y_train)
ada_cv = ada.best_estimator_
print(ada_cv)

y_pred = ada.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
ada_mis = 1- accuracy_score(y_test, y_pred)
print(ada_mis)



### Emsamble.




### ~N  Dscriminant analysis


### Model evaluation

