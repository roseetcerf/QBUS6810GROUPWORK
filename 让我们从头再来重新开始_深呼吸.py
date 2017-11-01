# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 18:25:54 2017

@author: Administrator
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


df = pd.read_csv("Clothing_Store.csv") # load CSV file
df.head()

df.shape 



## data cleaning
X = df.iloc[:, :-1]

phone = pd.get_dummies(X['VALPHON'], drop_first=True) # Get dummy
phone = phone.rename(columns = {'Y':'VALPHON'})

X = X.drop('VALPHON', axis=1)
X = X.drop('HHKEY', axis=1)
X = X.drop('ZIP_CODE', axis=1)

# Cleaned Variables & target
X = pd.concat([X, phone], axis=1)
y = df.iloc[:,-1]


# missing value check
import scipy as sp
sp.sum(sp.isnan(X)) 
sp.sum(sp.isnan(y)) 

## data understanding & preparation (EDA, graph, descript)

print("Proportion of responders: {0:.2f}%".format(float(sum(y))/float(len(y))*100))

# Clustype varaible
from collections import Counter
style = Counter(X['CLUSTYPE'])
style_prob = [(i, style[i] / len(X['CLUSTYPE']) * 100.0) for i, count in style.most_common()]
style_prob = pd.DataFrame(style_prob)
style_prob = pd.DataFrame({'lifestyle':style_prob[0], 'prob': style_prob[1]})
style_prob.head()

# skew + kurtosis
descrip = X.describe()
descrip.loc['skew', :] = X.skew()
descrip.loc['kurt', :] = X.kurt()
print(descrip.round(3))

def hist(series):
    fig,ax=plt.subplots(figsize=(9,7))
    sns.distplot(series,ax=ax,hist_kws={'alpha':0.9, 'edgecolor':'black','color':sns.color_palette('Blues', n_colors=2)[-1]},
                                        kde_kws={'color':'black','alpha':0.7})
    return fig, ax

fig = plt.figure()
hist(X['HI'])
sns.despine()   
plt.show()
fig.savefig('Product uniformity.pdf')      #### [存不下来啊？！！？！,自定义的function？]


## Numerical variables
# Normalization
from sklearn import preprocessing
X_num = X.drop(['CC_CARD','VALPHON','WEB'], axis=1)


# no '0' - natural log transformation
X_num1 = X_num.loc[:, (X_num != 0).all(axis=0)]
X_log = np.log(X_num1)

# contain '0' - square root transformation 
X_num0 = X_num.drop(X_num1, axis=1)
X_sqrt = np.sqrt(X_num0)


X_num = pd.concat([X_log, X_sqrt], axis=1)


# test with above example variable
fig = plt.figure()
hist(X_num['HI'])
sns.despine()   
plt.show()
plt.savefig('Product uniformity.pdf')      #### [存不下来啊？！！？！,自定义的function？]


# 15 closing varaibles - choose one as example
plt.figure()
hist(X['PJACKETS'])
hist(X_num['PJACKETS'])
sns.despine() 
plt.show()
fig.savefig('PJACKETS.pdf')      #### [存不下来啊？！！？！,自定义的function？]

jacket = Counter(X_num['PJACKETS'])
print("Proportion of not buy jackets: {0:.2f}%".format(jacket[0]/len(X_num['PJACKETS']) * 100.0))


## Standerdize
# differences in std
descrip = X_num.describe()
print("Std of last 6mos spend: {0:.2f}".format(np.std(X_num['SMONSPEND'])))
print("Std of coupon spend: {0:.2f}".format(np.std(X_num['COUPONS'])))

# z-score
from sklearn import preprocessing
X_scale = preprocessing.scale(X_num)
X_scale = pd.DataFrame(X_scale, columns = X_num.columns, index = X_num.index)

# one variable to prove [6mos spend]
plt.figure()
hist(X_scale['SMONSPEND'])
sns.despine() 
plt.show()
fig.savefig('SMONSPEND.pdf')      #### [存不下来啊？！！？！,自定义的function？]

## derive new variables.
# amount spent
X_spent = X_scale[['TMONSPEND', 'OMONSPEND', 'SMONSPEND']]

X_spent23 = X_scale['TMONSPEND']-X_scale['OMONSPEND']
X_spent23 = pd.DataFrame({'2-3MONSPEND': X_spent23})

X_spent46 = X_scale['SMONSPEND']-X_scale['TMONSPEND']
X_spent46 = pd.DataFrame({'4-6MONSPEND': X_spent46})

# replace TMONSPEND & SMONSPEND to 23spent & 46spent
X_spent_n= pd.concat([X_spent['OMONSPEND'], X_spent23, X_spent46], axis=1)
X_scale = X_scale.drop(X_spent, axis=1)
X_scale = pd.concat([X_scale, X_spent_n], axis=1)

# functional relationship
X_func = X_scale[['FRE', 'MON', 'AVRG']]
X_func_corr = X_func.corr()
#return this to below~~~~


## Rough: Relationships b/w the Predictors and the Response
# merge categoritical variable back
X_cal = X[['CC_CARD', 'VALPHON', 'WEB']]
X = pd.concat([X_scale, X_cal], axis = 1)

corr_y = X.corrwith(y).sort_values(ascending = False)
corr_y_abs = X.corrwith(y).abs().sort_values(ascending = False)
print(corr_y.head(8))
print(corr_y_abs.head(8))


# highest corr plot 
lifevisit= pd.concat([X['LTFREDAY'], y], axis =1)
lifevisit_1 = lifevisit[lifevisit['RESP'] !=0]
lifevisit_0= lifevisit[lifevisit['RESP'] ==0]

fig, ax = plt.subplots(figsize=(9,7))
sns.distplot(lifevisit_0['LTFREDAY'],bins=None, ax=ax, kde=False, label='no response')
sns.distplot(lifevisit_1['LTFREDAY'],bins=None, ax=ax, kde=False, label='responsd')
plt.legend()
plt.ylabel("Count")
plt.title("LTFREDAY w/ differnt response")
plt.savefig('LTFREDAY with differnt response.pdf')

# normalize histograph
fig, ax = plt.subplots(figsize=(9,7))
sns.distplot(lifevisit_0['LTFREDAY'],bins=None, ax=ax, norm_hist = True, kde=False, label='no response')
sns.distplot(lifevisit_1['LTFREDAY'],bins=None, ax=ax, norm_hist = True, kde=False, label='responsd')
plt.legend()
plt.ylabel("Percentage")
plt.title("Norm LTFREDAY w/ differnt response")
plt.savefig('Norm LTFREDAY with differnt response.pdf')



#other high corrs    [这他妈y轴为什么不是sum在1？？？？]
high_corr = X[['FRE', 'STYLES', 'RESPONDED', 'MON', 'CLASSES', 'COUPONS', 'FREDAYS','LTFREDAY']]
high_corr = pd.concat([high_corr, y], axis=1)
high_corr_1 = high_corr[high_corr['RESP'] !=0]
high_corr_0 = high_corr[high_corr['RESP'] ==0]


# Set up the matplotlib figure
f, axes = plt.subplots(2, 4, figsize=(10,10))
sns.despine(left=True)

# subplots
sns.distplot(high_corr_0['FRE'],bins=None, ax=axes[0,0], norm_hist = True, kde=False, label='no response')
sns.distplot(high_corr_1['FRE'],bins=None, ax=axes[0,0], norm_hist = True, kde=False, label='responsd')
axes[0,0].legend(loc="upper right")

sns.distplot(high_corr_0['STYLES'],bins=None, ax=axes[0,1], norm_hist = True, kde=False, label='no response')
sns.distplot(high_corr_1['STYLES'],bins=None, ax=axes[0,1], norm_hist = True, kde=False, label='responsd')
axes[0,1].legend(loc="upper right")

sns.distplot(high_corr_0['RESPONDED'],bins=None, ax=axes[0,2], norm_hist = True, kde=False, label='no response')
sns.distplot(high_corr_1['RESPONDED'],bins=None, ax=axes[0,2], norm_hist = True, kde=False, label='responsd')
axes[0,2].legend(loc="upper right")

sns.distplot(high_corr_0['MON'],bins=None, ax=axes[0,3], norm_hist = True, kde=False, label='no response')
sns.distplot(high_corr_1['MON'],bins=None, ax=axes[0,3], norm_hist = True, kde=False, label='responsd')
axes[0,3].legend(loc="upper right")

sns.distplot(high_corr_0['CLASSES'],bins=None, ax=axes[1,0], norm_hist = True, kde=False, label='no response')
sns.distplot(high_corr_1['CLASSES'],bins=None, ax=axes[1,0], norm_hist = True, kde=False, label='responsd')
axes[1,0].legend(loc="upper right")

sns.distplot(high_corr_0['COUPONS'],bins=None, ax=axes[1,1], norm_hist = True, kde=False, label='no response')
sns.distplot(high_corr_1['COUPONS'],bins=None, ax=axes[1,1], norm_hist = True, kde=False, label='responsd')
axes[1,1].legend(loc="upper right")

sns.distplot(high_corr_0['FREDAYS'],bins=None, ax=axes[1,2], norm_hist = True, kde=False, label='no response')
sns.distplot(high_corr_1['FREDAYS'],bins=None, ax=axes[1,2], norm_hist = True, kde=False, label='responsd')
axes[1,2].legend(loc="upper right")

sns.distplot(high_corr_0['LTFREDAY'],bins=None, ax=axes[0,0], norm_hist = True, kde=False, label='no response')
sns.distplot(high_corr_1['LTFREDAY'],bins=None, ax=axes[0,0], norm_hist = True, kde=False, label='responsd')
axes[1,3].legend(loc="upper right")


axes[0,0].set_ylabel('Percentage')
axes[1,0].set_ylabel('Percentage')
plt.suptitle('Norm high_corr variables w/ responses', fontsize=16)
plt.savefig('Norm high_corr with differnt response.pdf')



# Norm privious closing percentage varaible w. response [这他妈y轴为什么不是sum在1？？？？]
jacket = pd.concat([X['PJACKETS'], y], axis=1)
jacket_1 = jacket[jacket['RESP'] !=0]
jacket_0= jacket[jacket['RESP'] ==0]

#plot
fig, ax = plt.subplots(figsize=(9,7))
sns.distplot(jacket_0['PJACKETS'],bins=None, ax=ax, norm_hist = True, kde=False, label='no response')
sns.distplot(jacket_1['PJACKETS'],bins=None, ax=ax, norm_hist = True, kde=False, label='responsd')
plt.legend()
plt.ylabel("Percentage")
plt.title("Norm PJACKETS w/ differnt response")
plt.savefig('Norm PJACKETS with differnt response.pdf')


# Norm Uniformity w/ response
uni = pd.concat([X['HI'], y], axis=1)
uni_1 = uni[uni['RESP'] !=0]
uni_0= uni[uni['RESP'] ==0]

#plot
fig, ax = plt.subplots(figsize=(9,7))
sns.distplot(uni_0['HI'],bins=None, ax=ax, norm_hist = True, kde=False, label='no response')
sns.distplot(uni_1['HI'],bins=None, ax=ax, norm_hist = True, kde=False, label='responsd')
plt.legend()
plt.ylabel("Percentage")
plt.title("Norm uniformity w/ differnt response")
plt.savefig('Norm uniformity with differnt response.pdf')


### correlation b/w variables -multicollinearity
corr_matrix = X.corr()
corr_index = corr_matrix.index

# high corr 
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if corr_matrix.iloc[i, j] >= 0.8:
            if corr_index[i] != corr_index[j]:
                print('high corr exist(+ve): ' +corr_index[i]+ ',' +corr_index[j])
                print(corr_matrix.iloc[i, j].round(2))
        if corr_matrix.iloc[i, j] <= -0.8:  
            if corr_index[i] != corr_index[j]:
                print('high corr exist(-ve): ' +corr_index[i]+ ',' +corr_index[j])
                print(corr_matrix.iloc[i, j].round(2))


# scatter plot -example
fig = plt.figure()
plt.scatter(X['MAILED'], X['PROMOS'])
plt.xlabel("MAILED")
plt.ylabel("PROMOS")
plt.title("MAILED vs. PROMOS")
plt.savefig("MAILED vs PROMOS.pdf")
 
fig = plt.figure()
plt.scatter(X['CLASSES'], X['HI'])
plt.xlabel("CLASSES")
plt.ylabel("Uniformoity")
plt.title("CLASSES vs. UNIFORMROITY")
plt.savefig("CLASSES vs. UNIFORMROITY.pdf")


# Jacket on flags varaibles & response [sample]
jacket_flags = pd.concat([jacket, X_cal], axis =1)

# prop w/ card + respond
cardj = jacket_flags[(jacket_flags['PJACKETS'] != 0) & (jacket_flags['RESP'] !=0) & (jacket_flags['CC_CARD'] !=0)]
cardj_prob = cardj['PJACKETS'].count()/len(y) *100
print("+ve response on Card for jacket buyer : {0:.2f}%".format(cardj_prob))                    

phonej = jacket_flags[(jacket_flags['PJACKETS'] != 0) & (jacket_flags['RESP'] !=0) & (jacket_flags['VALPHON'] !=0)]
phonej_prob = phonej['PJACKETS'].count()/len(y) *100
print("+ve response on Phone for jacket buyer : {0:.2f}%".format(phonej_prob))                    

webj = jacket_flags[(jacket_flags['PJACKETS'] != 0) & (jacket_flags['RESP'] !=0) & (jacket_flags['WEB'] !=0)]
webj_prob = webj['PJACKETS'].count()/len(y) *100
print("+ve response on Web for jacket buyer: {0:.2f}%".format(webj_prob))                    
 

## derive other flag variables

# variables contain categorical meaning
types = df[['PSWEATERS','PKNIT_TOPS','PKNIT_DRES','PBLOUSES','PJACKETS','PCAR_PNTS','PCAS_PNTS','PSHIRTS','PDRESSES','PSUITS','POUTERWEAR','PJEWELRY','PFASHION','PLEGWEAR','PCOLLSPND']]
spent = df[['AMSPEND','PSSPEND','CCSPEND','AXSPEND','OMONSPEND','TMONSPEND','SMONSPEND','PREVPD']]
others = df[['RESPONSERATE','PERCRET']]

# binary those variables as 0 is no response, 1 is response
types= types*999999
types = types.clip_upper(1)

spent = spent*999999
spent = spent.clip_upper(1)

others = others*999999
others = others.clip_upper(1)

# merge new flag variables together.
flags = pd.concat([types, spent, others], axis=1)
flags.columns = ['Flag_'] + flags.columns 

# merge with X
X = pd.concat([X, flags], axis=1)

# =============================================================================
# =============================================================================



##### Modeling 
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, precision_score, recall_score


# Split dataset 
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state = 460475789)

X_train0 = X_train
y_train0 = y_train
# Handle imbalanced dataset - on train data
from sklearn.utils import resample
df1 = pd.concat([X_train, y_train], axis=1)

df1_majo = df1[df1['RESP']==0]
df1_mino = df1[df1['RESP']==1]

# type1: minority upsample
df1_mino_upsampled = resample(df1_mino, replace=True, n_samples=(len(df1_majo)))

df1_upsampled = pd.concat([df1_majo, df1_mino_upsampled])

df1_upsampled['RESP'].value_counts()
X_train1 = df1_upsampled.iloc[:, :-1]
y_train1 = df1_upsampled.iloc[:,-1]

# type2: majority downsample
df1_majo_downsampled = resample(df1_majo, replace=False, n_samples=(len(df1_mino)))

df1_downsampled = pd.concat([df1_mino, df1_majo_downsampled])
X_train2 = df1_downsampled.iloc[:, :-1]
y_train2 = df1_downsampled.iloc[:,-1]

# after test with all the models with unbalanced, or two differnt weights methods, we choose type 1
X_train = X_train1
y_train = y_train1


## baseline performance
num_response0 = y_test[y_test ==0].count()
num_response1 = y_test[y_test ==1].count()
print("testset clients num: {0}".format(y_test.count()))
print("testset +ve response: {0}".format(num_response1))
print("testset -ve response: {0}".format(num_response0))

##### 然后等一下再弄那个表格的后半部Table7.10



# =============================================================================
### Model1: Naive bayes 
from sklearn.naive_bayes import GaussianNB, BernoulliNB
X_train_num = X_train.iloc[:, :45]
X_train_cat = X_train.iloc[:, 45:]

X_test_num = X_test.iloc[:, :45]
X_test_cat = X_test.iloc[:, 45:]

# combine GNB & BNB.
gnb = GaussianNB().fit(X_train_num, np.ravel(y_train))
gnb_prob_train = gnb.predict_proba(X_train_num)
gnb_prob_test = gnb.predict_proba(X_test_num)

bnb = BernoulliNB().fit(X_train_cat, np.ravel(y_train))
bnb_prob_train = bnb.predict_proba(X_train_cat)
bnb_prob_test = bnb.predict_proba(X_test_cat)

nb_train = np.hstack((gnb_prob_train, bnb_prob_train))
nb_test = np.hstack((gnb_prob_test, bnb_prob_test))

# combined NB model fit
nb = GaussianNB().fit(nb_train, np.ravel(y_train))

# predict
nb_pred = nb.predict(nb_test)
nb_prob = nb.predict_proba(nb_test)

nb_confusion = confusion_matrix(y_test, nb_pred)
nb_mis = (1- accuracy_score(y_test, nb_pred)).round(3)
nb_se = np.sqrt(nb_mis*(1- nb_mis)/len(y_test)).round(3)
nb_sensi = nb_confusion[1,1]/np.sum(nb_confusion[1,:]).round(3)
nb_speci = nb_confusion[0,0]/np.sum(nb_confusion[0,:]).round(3)
nb_auc = roc_auc_score(y_test, nb_prob[:,1]).round(3)
nb_precision = precision_score(y_test, nb_pred).round(3)


columns = ['Combined Naive Bayes']
rows = ['Error rate', 'SE', 'Sensitivity', 'Specificity', 'AUC', 'Precision']

nb_score = pd.DataFrame([nb_mis, nb_se, nb_sensi, nb_speci, nb_auc, nb_precision], columns = columns, index = rows)
print(nb_score)

# Confidence interval
n_iterations = 10000
n_size = len(y_test)
test = nb_test

accy = list()
for i in range(n_iterations):
    test = resample(test, n_samples=n_size)
    y_prob = nb.predict_proba(test)
    score = accuracy_score(y_test, y_prob)
    accy.append(score)

alpha = 0.95
p = ((1-alpha)/2) * 100
ci_low = np.percentile(accy, p)
q = (1-((1-alpha)/2)) * 100
ci_high = np.percentile(accy, q)

print('the %.0f%% confidence interval is (%.2f, %.2f)' % (alpha*100, ci_low*100, ci_high*100))
#
# =============================================================================
##E Model2-1: LogisticRegression
from sklearn.linear_model import LogisticRegression
logit = LogisticRegression()
logit.fit(X_train, y_train)

logit_pred = logit.predict(X_test)
logit_prob = logit.predict_proba(X_test)


logit_confusion = confusion_matrix(y_test, logit_pred)
logit_mis = (1- accuracy_score(y_test, logit_pred)).round(3)
logit_se = np.sqrt(logit_mis*(1- logit_mis)/len(y_test)).round(3)
logit_sensi = logit_confusion[1,1]/np.sum(logit_confusion[1,:]).round(3)
logit_speci = logit_confusion[0,0]/np.sum(logit_confusion[0,:]).round(3)
logit_auc = roc_auc_score(y_test, logit_prob[:,1]).round(3)
logit_precision = precision_score(y_test, logit_pred).round(3)

columns = ['Logistic']
rows = ['Error rate', 'SE', 'Sensitivity', 'Specificity', 'AUC', 'Precision']

logit_score = pd.DataFrame([logit_mis, logit_se, logit_sensi, logit_speci, logit_auc, logit_precision], columns = columns, index = rows)



# Confidence interval
n_iterations = 10000
n_size = len(y_test)
test = X_test

accy = list()
for i in range(n_iterations):
    test = resample(test, n_samples=n_size)
    y_prob = lotig.predict_proba(test)
    score = accuracy_score(y_test, y_prob)
    accy.append(score)

alpha = 0.95
p = ((1-alpha)/2) * 100
ci_low = np.percentile(accy, p)
q = (1-((1-alpha)/2)) * 100
ci_high = np.percentile(accy, q)

print('the %.0f%% confidence interval is (%.2f, %.2f)' % (alpha*100, ci_low*100, ci_high*100))
#
#
# =============================================================================
### Model 2-2: LogisticRegressionCV (L1 - lasso)
from sklearn.linear_model import LogisticRegressionCV
logit_l1 = LogisticRegressionCV(penalty = 'l1', solver = 'liblinear', cv=5)
logit_l1.fit(X_train, y_train)  # fit training set

# Predict
logit_l1_pred = logit_l1.predict(X_test)   # predit on test set
logit_l1_prob = logit_l1.predict_proba(X_test)

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


# Confidence interval
n_iterations = 10000
n_size = len(y_test)
test = X_test

accy = list()
for i in range(n_iterations):
    test = resample(test, n_samples=n_size)
    y_prob = lotig_l1.predict_proba(test)
    score = accuracy_score(y_test, y_prob)
    accy.append(score)

alpha = 0.95
p = ((1-alpha)/2) * 100
ci_low = np.percentile(accy, p)
q = (1-((1-alpha)/2)) * 100
ci_high = np.percentile(accy, q)

print('the %.0f%% confidence interval is (%.2f, %.2f)' % (alpha*100, ci_low*100, ci_high*100))
#
#
# =============================================================================
### Model 2-3: LogisticRegressionCV (L2 - Ridge)
logit_l2 = LogisticRegressionCV(penalty = 'l2', cv=5)
logit_l2.fit(X_train, y_train)  # fit training set


# Predict
logit_l2_pred = logit_l2.predict(X_test)   # predit on test set
logit_l2_prob = logit_l2.predict_proba(X_test)

logit_l2_confusion = confusion_matrix(y_test, logit_l2_pred)
logit_l2_mis = (1- accuracy_score(y_test, logit_l2_pred)).round(3)
logit_l2_se = np.sqrt(logit_l2_mis*(1- logit_l2_mis)/len(y_test)).round(3)
logit_l2_sensi = logit_l2_confusion[1,1]/np.sum(logit_l2_confusion[1,:]).round(3)
logit_l2_speci = logit_l2_confusion[0,0]/np.sum(logit_l2_confusion[0,:]).round(3)
logit_l2_auc = roc_auc_score(y_test, logit_l2_prob[:,1]).round(3)
logit_l2_precision = precision_score(y_test, logit_l2_pred).round(3)

columns = ['L2 regularised']
rows = ['Error rate', 'SE', 'Sensitivity', 'Specificity', 'AUC', 'Precision']

logit_l2_score = pd.DataFrame([logit_l2_mis, logit_l2_se, logit_l2_sensi, logit_l2_speci, logit_l2_auc, logit_l2_precision], columns = columns, index = rows)

# Confidence interval
n_iterations = 10000
n_size = len(y_test)
test = X_test

accy = list()
for i in range(n_iterations):
    test = resample(test, n_samples=n_size)
    y_prob = lotig_l2.predict_proba(test)
    score = accuracy_score(y_test, y_prob)
    accy.append(score)

alpha = 0.95
p = ((1-alpha)/2) * 100
ci_low = np.percentile(accy, p)
q = (1-((1-alpha)/2)) * 100
ci_high = np.percentile(accy, q)

print('the %.0f%% confidence interval is (%.2f, %.2f)' % (alpha*100, ci_low*100, ci_high*100))

# =============================================================================
### Model 3-1 Gaussian Discriminant analysis [numerical only]  - LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
X_train_num = X_train.iloc[:, :45]
X_test_num = X_test.iloc[:, :45]

lda = LinearDiscriminantAnalysis()
lda.fit(X_train_num, y_train)

# predict
lda_pred = lda.predict(X_test_num)
lda_prob = lda.predict_proba(X_test_num)

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

# Confidence interval
n_iterations = 10000
n_size = len(y_test)
test = X_test_num

accy = list()
for i in range(n_iterations):
    test = resample(test, n_samples=n_size)
    y_prob = lda.predict_proba(test)
    score = accuracy_score(y_test, y_prob)
    accy.append(score)

alpha = 0.95
p = ((1-alpha)/2) * 100
ci_low = np.percentile(accy, p)
q = (1-((1-alpha)/2)) * 100
ci_high = np.percentile(accy, q)

print('the %.0f%% confidence interval is (%.2f, %.2f)' % (alpha*100, ci_low*100, ci_high*100))

# =============================================================================
### Model 3-2: QDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis 
X_train_num = X_train.iloc[:, :45]
X_test_num = X_test.iloc[:, :45]

qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train_num, y_train)

# predict
qda_pred = qda.predict(X_test_num)
qda_prob = qda.predict_proba(X_test_num)

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

# Confidence interval
n_iterations = 10000
n_size = len(y_test)
test = X_test_num

accy = list()
for i in range(n_iterations):
    test = resample(test, n_samples=n_size)
    y_prob = qda.predict_proba(test)
    score = accuracy_score(y_test, y_prob)
    accy.append(score)

alpha = 0.95
p = ((1-alpha)/2) * 100
ci_low = np.percentile(accy, p)
q = (1-((1-alpha)/2)) * 100
ci_high = np.percentile(accy, q)

print('the %.0f%% confidence interval is (%.2f, %.2f)' % (alpha*100, ci_low*100, ci_high*100))


# =============================================================================
### Model 3-2 Regulized QDA
from sklearn.model_selection import GridSearchCV
X_train_num = X_train.iloc[:, :45]
X_test_num = X_test.iloc[:, :45]

alphas = np.linspace(0, 1, 21)
param_grid = {'reg_param': alphas}  
gscv = GridSearchCV(QuadraticDiscriminantAnalysis(), param_grid, cv = 5, scoring = 'accuracy')
gscv.fit(X_train_num, y_train)

qda_reg = gscv.best_estimator_
qda_reg.fit(X_train_num, y_train)

# predict
qda_reg_pred = qda_reg.predict(X_test_num)
qda_reg_prob = qda_reg.predict_proba(X_test_num)

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

# Confidence interval
n_iterations = 10000
n_size = len(y_test)
test = X_test_num

accy = list()
for i in range(n_iterations):
    test = resample(test, n_samples=n_size)
    y_prob = qda_reg.predict_proba(test)
    score = accuracy_score(y_test, y_prob)
    accy.append(score)

alpha = 0.95
p = ((1-alpha)/2) * 100
ci_low = np.percentile(accy, p)
q = (1-((1-alpha)/2)) * 100
ci_high = np.percentile(accy, q)

print('the %.0f%% confidence interval is (%.2f, %.2f)' % (alpha*100, ci_low*100, ci_high*100))


# =============================================================================
#### Model 4 - KNN
from sklearn import neighbors
from sklearn.neighbors import KNeighborsClassifier


#Gridsearch, model fit
k_grid = {"n_neighbors": np.arange(1, 32, 1)}
knn = GridSearchCV(neighbors.KNeighborsClassifier(), k_grid, cv = 5, n_jobs=-1)
knn.fit(X_train, y_train)
knn_cv = knn.best_estimator_
knn_cv.fit(X_train, y_train)

# predict
knn_pred = knn_cv.predict(X_test)
knn_prob = knn_cv.predict_proba(X_test)

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


# Confidence interval
n_iterations = 10000
n_size = len(y_test)
test = X_test

accy = list()
for i in range(n_iterations):
    test = resample(test, n_samples=n_size)
    y_prob = knn_cv.predict_proba(test)
    score = accuracy_score(y_test, y_prob)
    accy.append(score)

alpha = 0.95
p = ((1-alpha)/2) * 100
ci_low = np.percentile(accy, p)
q = (1-((1-alpha)/2)) * 100
ci_high = np.percentile(accy, q)

print('the %.0f%% confidence interval is (%.2f, %.2f)' % (alpha*100, ci_low*100, ci_high*100))

# =============================================================================
### Model 5- Decision Tree
from sklearn import tree
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier 

# base on max_depth & min_samples_leaf
pipeline = Pipeline([('clf',DecisionTreeClassifier(criterion='entropy'))])
parameters = {'clf__max_depth': np.arange(2, 50, 1), 'clf__min_samples_leaf': [1,5,10,20]}
grid_search = GridSearchCV(pipeline, parameters, cv=5)
grid_search.fit(X_train, y_train)
tree = grid_search.best_estimator_
tree.fit(X_train, y_train)

# Predict
tree_pred = tree.predict(X_test)
tree_prob = tree.predict_proba(X_test)

tree_confusion = confusion_matrix(y_test, tree_pred)
tree_mis = (1- accuracy_score(y_test, tree_pred)).round(3)
tree_se = np.sqrt(tree_mis*(1- tree_mis)/len(y_test)).round(3)
tree_sensi = tree_confusion[1,1]/np.sum(tree_confusion[1,:]).round(3)
tree_speci = tree_confusion[0,0]/np.sum(tree_confusion[0,:]).round(3)
tree_auc = roc_auc_score(y_test, tree_prob[:,1]).round(3)
tree_precision = precision_score(y_test, tree_pred).round(3)

columns = ['Decision Tree_depths']
rows = ['Error rate', 'SE', 'Sensitivity', 'Specificity', 'AUC', 'Precision']

tree_score = pd.DataFrame([tree_mis, tree_se, tree_sensi, tree_speci, tree_auc, tree_precision], columns = columns, index = rows)


# Confidence interval
n_iterations = 10000
n_size = len(y_test)
test = X_test

accy = list()
for i in range(n_iterations):
    test = resample(test, n_samples=n_size)
    y_prob = tree.predict_proba(test)
    score = accuracy_score(y_test, y_prob)
    accy.append(score)

alpha = 0.95
p = ((1-alpha)/2) * 100
ci_low = np.percentile(accy, p)
q = (1-((1-alpha)/2)) * 100
ci_high = np.percentile(accy, q)

print('the %.0f%% confidence interval is (%.2f, %.2f)' % (alpha*100, ci_low*100, ci_high*100))


## =============================================================================
### Ensamble
from sklearn.ensemble import VotingClassifier
# Voting Classifier
eclf = VotingClassifier(estimators = [('Logit', logit_l2), ('Naive Bayes', nb), ('kNN', knn), ('Decision Tree', tree), ('Discriminant Analysis', lda)], voting = 'hard')

eclf.fit(X_train, y_train)

#predict
eclf_pred= eclf.predict(X_test)
eclf_prob = eclf.predict_proba(X_test)

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

# Confidence interval
n_iterations = 10000
n_size = len(y_test)
test = X_test

accy = list()
for i in range(n_iterations):
    test = resample(test, n_samples=n_size)
    y_prob = eclf.predict_proba(test)
    score = accuracy_score(y_test, y_prob)
    accy.append(score)

alpha = 0.95
p = ((1-alpha)/2) * 100
ci_low = np.percentile(accy, p)
q = (1-((1-alpha)/2)) * 100
ci_high = np.percentile(accy, q)

print('the %.0f%% confidence interval is (%.2f, %.2f)' % (alpha*100, ci_low*100, ci_high*100))
# =============================================================================
# =============================================================================

### Model Evalaution
summary = pd.concat([nb_score, logit_score, logit_l1_score, logit_l2_score, tree_score, lda_score, qda_score, qda_reg_score, eclf_score], axis=1)                
summary.to_csv('Model evaluation_summary.csv')



print(nb_confusion)
print(logit_confusion)
print(logit_l1_confusion)
print(logit_l2_confusion)
print(lda_confusion)
print(qda_confusion)
print(qda_reg_confusion)
print(knn_confusion)
print(tree_confusion)
print(eclf_confusion)




# ROC curve
#palette = ['#1F77B4', '#FF7F0E', '#2CA02C', '#980002', '#9ffeb0', '#516572', '#4b006e', '#DB2728', '#9467BD', '#95A5A6', '#34495E', '#137e6d']

palette = ['#1F77B4', '#FF7F0E', '#2CA02C', '#DB2728', '#9467BD', '#9467BD']

from sklearn.metrics import roc_curve

labels=['Naive Bayes', 'L2 regularised', 'KNN', 'Decision Tree', 'Ensambled', 'LDA']
methods=[nb, logit_l2, knn, tree, eclf, lda]

fig, ax= plt.subplots(figsize=(9,6))

for i, method in enumerate(methods):
    if i < 5:
        y_prob = method.predict_proba(X_test)
    else:
        y_prob = method.predict_proba(X_test_num)
        
    fpr, tpr, _ = roc_curve(y_test, y_prob[:,1])
    auc = roc_auc_score(y_test, y_prob[:,1])
    ax.plot(1-fpr, tpr, label=labels[i] + ' (AUC = {:.3f})'.format(auc), color = palette[i])

    
ax.set_xlabel('Specificity')
ax.set_ylabel('Sensitivity')
ax.set_title('ROC curves', fontsize=14)
plt.legend(fontsize=13)
plt.show()
fig.savefig('ROC.pdf')

# feature_importance
from statlearning import plot_feature_importance
plot_feature_importance(tree, X_train)
plt.show()
fig.savefig('best features.pdf')

## =============================================================================
#
## Random Foreast [bagging] [have a try]
#from sklearn.ensemble import RandomForestClassifier 
## Gridsearch
#model = RandomForestClassifier(n_estimators=1000)
#tuning_parameters = [{'min_samples_leaf': [1,5,10],'max_features': list(np.arange(2,len(X_train))),}]
#
## the n_jobs option enables parallel processing
#rfg = GridSearchCV(model, tuning_parameters, cv=5, return_train_score=False, n_jobs=-1)
#rfg.fit(X_train, y_train)
#rf = rfg.best_estimator_
#rf.fit(X_train, y_train)
#
#
## Predict
#rf_pred = rf.predict(X_test)
#rf_prob = rf.predict_proba(X_test)
#
#rf_confusion = confusion_matrix(y_test, rf_pred)
#rf_mis = (1- accuracy_score(y_test, rf_pred)).round(3)
#rf_se = np.sqrt(rf_mis*(1- rf_mis)/len(y_test)).round(3)
#rf_sensi = rf_confusion[1,1]/np.sum(rf_confusion[1,:]).round(3)
#rf_speci = rf_confusion[0,0]/np.sum(rf_confusion[0,:]).round(3)
#rf_auc = roc_auc_score(y_test, rf_prob[:,1]).round(3)
#rf_precision = precision_score(y_test, rf_pred).round(3)
#
#columns = ['Decision Tree_features']
#rows = ['Error rate', 'SE', 'Sensitivity', 'Specificity', 'AUC', 'Precision']
#
#rf_score = pd.DataFrame([rf_mis, rf_se, rf_sensi, rf_speci, rf_auc, rf_precision], columns = columns, index = rows)

