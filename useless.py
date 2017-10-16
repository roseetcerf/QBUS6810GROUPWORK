# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 23:38:54 2017

@author: Administrator
"""

# Fit and compute predictions
logit = LogisticRegression()
logit.fit(X_train, y_train)
y_pred = logit.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
logit_mis = 1- accuracy_score(y_test, y_pred)
print(logit_mis)



logit_l2 = LogisticRegressionCV(penalty = 'l2')
logit_l2.fit(X_train, y_train)
y_pred = logit_l2.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
logit_l2_mis = 1- accuracy_score(y_test, y_pred)
print(logit_l2_mis)


for i in range(50):
    hist(X_train[X[i]])
    sns.despine()
    plt.show()
###    
#
#for i in range(50):
#    sns.regplot(X_train[X[i]], y_trian, color=sns.color_palette('Blues')[-1] ci = None, logistic=True, y_jitter = 0.05,
#                scatter_kws={'s': 25, 'color':sns.color_palette('Blues')[-1], 'alpha': .5})
#    sns.despine()
#    plt.show()

