# QBUS6810GroupWork - Task 2

#### Always Open "Workbook" & Trace detailed edit.
#### Always Edit "README.md" for timeline record.

## ~xy. 10/16:
- "Workbook" 更新细节
- "PCA: 10个components"   
- “"Variable selection": Decition tree (4_maxdepth), Logistic Regression(15), Forward (20)
## ~Smile. 10/17: 
- "workbook" data understanding 和 data preparation 更新细节
- 疑问：是不是所有variable selection 方式前，都需要data standardization？原因：将所有的数据转化为同一个尺度。(Tutorial 10 用到了 data standardization)
## ~xy. 10/19:
- 是不是把part 3-5合在一起写比较好，以不同模型来分。然后在part6的时候进行统一比较比较方便。要不然很难摘的那么细？
- 然后，因为jupyter不好下载下来编辑，要不还是现在spyder上弄吧~~ 更新的时候，稍微说一下spyder上都改了什么。为了比较好找
- [数据]:data transform是放在一开始好还是保留在现在的位置。
- 然后，还都需要什么数据？对于不同的模型？希望用什么值来进行比较？
- ppt，tutorial跟那个写作的method上，有什么是特殊要注意的？
## ~Smile. 10/20
- 根据tutorial 10, transformation是在EDA之后，variable selection 之前，应该放在原位置就可以了吧？
- 今天问了stephen, variable selection的最终选择由CV score 来决定，不同模型的accuracy的比较按自己的偏好来，我问他RMSE可不可以，他说按自己的喜好就好啦。
- [regression spline]: 这周（11周）的tutorial regression spline,感觉可以用进我们的project。以下是咨询stephen的内容：regression splines 用在 feature engineering 的部分，如果有时间有精力的话，可以对所有低correlation（因为correlation描述的是线性关系，所以低correlation说明可能存在非线性关系）的variable（补充：跟小伙伴讨论觉得应该是所有numerical variables）进行测试。找出关系后，按照关系对raw data 进行处理(e.g.比如发现了一个变量中存在qudratic function,那么就将对应的变量的原数据做同样的处理），将全部处理完的新的data再去做variable selection。（不过总体感觉这个步骤好复杂很耗时间，我们可以讨论一下要不要做，做到什么程度。）
-raw data multicollinearity的问题，我们是否需要最开始做一个所有变量的 correlation matrix, 如果相互之间correlation高的话可能会存在multicollinearity，可以用interaction去解决。（见week3 lecture）
-[data transformation] week 3 讲了data transformation,举例也是用的direct marketing,但是它的transformation是用取log来完成的。我不确定我们是不是也要取log~
-week 3 的 lecture也讲到了categorical variables可以用到哪些图去做feature engineering,感觉我们也可以挑几个。
-今天出了通知说最后juypter的文件也需要提交，然后今天出了task2的评分标准。
           
## ~xy. 10/22
- 到roc curves了，出了model evaluation 的summary。 【Model evaluation_summary.csv】
- 还欠confidence interval
- summary中：除去了forward 【predict 出来的不是binary？？】， pca 【不在这里】，adaboost & emsamble 【跑不动，code在】。
- summary中：包含的有：logit, logit_l1, tree(max_depths), tree(max_features), knn, lda, qda, qda_reg.
- 上传了新的code.
- workbook跟readme功能互换（worbook为大框架更改，readme为细节更改）
- 【问题】assignment requiremnts 上的第3点：predictive power(用roc来说)，interpretability（用PCA来说？）。所以是说through different methods?

## ~xy. 10/23
- 加了Neural Network, Naive Bayes（Gaussian）. 【图：ROC.pdf】 【csv: Model evaluation_summary】
- 更新code

## ~Smile. 10/24
- 加入求personal expected profit 的code部分
- 在邮政官网找到direct mailing marketing的 postage的成本，比较保守取其中最高的一种类型是$1.550，加上打印成本assume $0.450(估计是彩页，比较偏高），一共$2.
postage来源：https://auspost.com.au/business/marketing-and-communications/bulk-mailouts/bulk-mail-options/acquisition-mail#tab2

- 更新data understanding feature engineering categorical variablles 部分 code
- Report 更新至Data understanding 结束。
- 【瓶颈】是否或者怎样对data做normaliztion的处理，目前知道的，至少PCA和Gussian Descriptive Analysis 需要normalization.(嘉庆的思路：只处理numerical variables, 处理完后和 categorical variables combine 再做接下来的分析。

## ~xy. 10/26
- normalize 了。

## ~xy. 10/27
-[code]加了weight (weight_Class = 'balanced')在logistic跟tree里，KNN需要再check一下。DA们和naive bayes不能设置。
" The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y))."
- [code] tree_depth 加了min_samples_leaf (还没跑~要跑一次看看depth 跟 leaf的数是多少，再用到fit里面). 可以再用一下gini看是不是比现在用entropy好。
- [code] 加个random foreast, 还没在evaluation那改，（没有跑，需要跑一下，大概比较久）【as bagging】
“After Importance: Let’s use the top 5 variables for creating a model. Also, we will modify the parameters of random forest model a little bit:”
- 【待处理】[code] 把emsamble改一改，用少一点的model来 【as boosting】 (还没改)
- add some references on references note.
- might need to use boxplot to identify the outlier. 
-【待处理】scaling是否要分别对待？貌似没必要，毕竟scale后，没影响的也不会怎样。 distance model：KNN， LDA,
《https://www.analyticsvidhya.com/blog/2016/07/practical-guide-data-preprocessing-python-scikit-learn/》

## ~xy. 10/28
-【待处理】 emsemble的时候，要选diversify的模型， “a random forest, a KNN, and a Naive Bayes”

## ~xy. 10/29
- 整套的EDA都出出来了，有部分地方需要一起讨论【尤其是对flag的那三个variable】，每一块用什么【sample】也可以讨论一下。
- 上传了code【python & jupyter】，pdf，图【待处理】（有的图没有lable，save清楚），别人组的code
- 【待处理】跑模型 P25起，解决ci
