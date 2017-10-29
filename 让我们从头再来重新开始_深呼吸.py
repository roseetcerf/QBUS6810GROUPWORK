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
high_corr = X[['FRE', 'STYLES', 'RESPONDED', 'MON', 'CLASSES', 'COUPONS', 'FREDAYS']]
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


# numerical problem [需要讨论一下，用什么办法处理flag，没太看明白web是怎么进行的逻辑， 但是结果说，反正都是minor的影响]
# Flag variables effect on spent
spent = df[['AMSPEND', 'PSSPEND','CCSPEND', 'AXSPEND','OMONSPEND', 'TMONSPEND','SMONSPEND','PREVPD']]
flags = X[['CC_CARD', 'VALPHON', 'WEB']]
spent_flags=pd.concat([spent, flags, y], axis=1)

spent_name = ['AMSPEND', 'PSSPEND','CCSPEND', 'AXSPEND','OMONSPEND', 'TMONSPEND','SMONSPEND','PREVPD']
flags_name = ['CC_CARD', 'VALPHON', 'WEB']
# prop w/ card + respond
prop_card = []
prop_phone = []
prop_web = []
for i in spent_name:
    card = spent_flags[(spent_flags[i] != 0) & (spent_flags['RESP'] !=0) & (spent_flags['CC_CARD'] !=0)]
    card = card[i].count()/len(y) *100
    phone = spent_flags[(spent_flags[i] != 0) & (spent_flags['RESP'] !=0) & (spent_flags['VALPHON'] !=0)]
    phone = phone[i].count()/len(y) *100
    web = spent_flags[(spent_flags[i] != 0) & (spent_flags['RESP'] !=0) & (spent_flags['WEB'] !=0)]
    web = web[i].count()/len(y) *100  
    prop_card.append(card)
    prop_phone.append(phone)
    prop_web.append(web)

prop_card = pd.DataFrame(np.array(prop_card), index = spent_name).rename(columns ={0: 'CC_CARD'})
prop_phone = pd.DataFrame(np.array(prop_phone), index = spent_name).rename(columns ={0: 'VALPHON'})
prop_web = pd.DataFrame(np.array(prop_web), index = spent_name).rename(columns ={0: 'WEB'})

prop_flags = pd.concat([prop_card, prop_phone, prop_web], axis=1)

#.sort_values(0, ascending=False)

# other way
flags.corrwith(y)


### correlation b/w variables -multicollinearity
corr_matrix = X.corr()

# getting the name of column
corr_col_0 = set()
corr_col_1 = set()
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if corr_matrix.iloc[i, j] >= 0.8:
            colname_1 = corr_matrix.columns[i] 
            corr_col_1.add(colname_1)
        if corr_matrix.iloc[i, j] <= -0.8:  
            colname_0 = corr_matrix.columns[i]            
            corr_col_0.add(colname_0)
                 
print(corr_col_1)
print(corr_col_0)

#corr_high = corr_matrix[['LTFREDAY', 'MAILED', 'CLASSES', 'STYLES', 'RESPONSERATE', 'HI']]
#
#corr_matrix_abs = corr_matrix.abs()
#
#
#a = ['LTFREDAY', 'MAILED', 'CLASSES', 'STYLES', 'RESPONSERATE']
#
#aa = corr_matrix.drop(corr_matrix.rows)
#
#corr_high.values <= 0.8
#pd.corr_high(data <0.8)
#
#
#a= corr_matrix.columns
#
#aaa= corr_matrix_abs.clip_lower(0.79)
#
#[set成str，然后删掉。]
#

# scatter plot -example
plt.scatter(X['MAILED'], X['PROMOS'])
plt.xlabel("MAILED")
plt.ylabel("PROMOS")
plt.title("MAILED vs. PROMOS")
plt.savefig("MAILED vs PROMOS.pdf")
 

plt.scatter(X['CLASSES'], X['HI'])
plt.xlabel("CLASSES")
plt.ylabel("Uniformoity")
plt.title("CLASSES vs. UNIFORMROITY")
plt.savefig("CLASSES vs. UNIFORMROITY.pdf")


# Jacket on flags varaibles & response [sample]
jacket_flags = pd.concat([jacket, flags], axis =1)

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
 