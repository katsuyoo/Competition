# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import scipy as sp

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore')

import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.svm import LinearSVC,SVC
from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier



import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression

def describe(X_train):
    # g = sns.FacetGrid(train_df, col='label')
    # g.map(plt.hist, 'tan')

    xgbc = XGBClassifier(n_estimators=100, learning_rate=0.1, min_child_weight=1)
    xgbc.fit(X, y)

    print np.array(zip(np.array(xgbc.feature_importances_),np.array(X_train.columns)))
    # print np.array(sorted(zip(np.array(xgbc.feature_importances_),np.array(X_train.columns)),key=lambda list1:list1[0]))



def score(y_test,y_pred):
    pre0=0.0
    act0=0.0
    pre_act=0.0
    n=len(y_pred)
    y_test=y_test.tolist()
    for i in range(n):
        if y_pred[i]==0:
            pre0+=1
        if y_test[i]==0:
            act0+=1
        if y_pred[i]==0 and y_test[i]==0:
            pre_act+=1
    p=(pre_act+1)/(pre0+1)
    r=(pre_act+1)/(act0+1)
    return 5*p*r/(2*p+3*r)*100

def cv(X,y):
    xgbc = XGBClassifier(n_estimators=120,learning_rate=0.1,min_child_weight=2)
    xgbc.fit(X,y)
    sc=make_scorer(score)
    print cross_val_score(xgbc,X,y,scoring=sc,cv=5,n_jobs=-1).mean()
    # print xgbc.feature_importances_
    print np.array(sorted(zip(np.array(xgbc.feature_importances_),np.array(X_train.columns)),key=lambda list1:list1[0]))


def standard_data(X):
    X_scaler=StandardScaler()
    X=X_scaler.fit_transform(X)
    return X

def parm_search(clf,params,X_train,y_train):
    print 'search......'
    sc=make_scorer(score)
    if __name__ == '__main__':
        gs=GridSearchCV(clf,params,cv=4,scoring=sc,n_jobs=-1)
        gs.fit(X_train,y_train)
        print gs.best_score_
        print gs.best_params_


def to_submission(y_pred,id):
    n=len(y_pred)
    print n
    res=[]
    for i in range(n):
        if y_pred[i] == 0:
            res.append(id[i])

    print len(res)
    res_df=pd.DataFrame({'id':res})
    res_df.to_csv('/home/frank/data/mouse/submission.csv',index=None)



train_df=pd.read_csv('/home/frank/data/mouse/train.csv')
test_df=pd.read_csv('/home/frank/data/mouse/test.csv')
# df=test_df[test_df['time_var']>1000000]
# print df['id']

# print train_df.columns

# 查看数据集统计信息
# print train_df.info()
# print test_df.info()
# describe(train_df)
# print train_df['tan'].value_counts()
# print train_df.describe()
# train_df[train_df['label']==1].describe().to_csv('/home/frank/data/mouse/describe1.csv')
# train_df[train_df['label']==0].describe().to_csv('/home/frank/data/mouse/describe0.csv')
# print train_df[train_df['label']==1].describe()
# print '-'*100
# print train_df[train_df['label']==0].describe()
# test_df.describe().to_csv('/home/frank/data/mouse/describe_test.csv')

X=train_df.drop(['label'],axis=1)
y=train_df['label']
# X_train,X_val,y_train,y_val=train_test_split(X,y,test_size=0.1,random_state=33)
X_train=X
y_train=y
# describe(X_train)

# print X.info()
X_test=test_df.drop(['id'],axis=1)
Id=test_df['id']


# X_train=standard_data(X_train)
# X_val=standard_data(X_val)
# X_test=standard_data(X_test)

# cv(X_train,y_train)

# 调参
# clf=LogisticRegression(penalty='l1')
# params={'penalty':['l1','l2']}
# params={'solver':['newton-cg','lbfgs','liblinear','sag']}
# params={'multi_class':['ovr','multinomial']}


# clf=XGBClassifier(n_estimators=120,learning_rate=0.1,min_child_weight=2)
# params={'n_estimators':np.arange(100,200,10)}
# params={'learning_rate': np.arange(0.1, 1, 0.1)}
# params={'min_child_weight':[1.9,2,2.1]}

# clf=SVC(C=1.0,random_state=33)
# params={"C":np.arange(1,2,1)}

# clf=RandomForestClassifier(n_estimators=80,random_state=330)
# params={'n_estimators':np.arange(10,100,10)}

# clf=GradientBoostingClassifier(n_estimators=120,learning_rate=0.1)
# params={'n_estimators':np.arange(100,200,10)}
# params={'learning_rate': np.arange(0.1, 1, 0.1)}

# clf=BaggingClassifier(n_estimators=40,random_state=101)
# params={'n_estimators':np.arange(10,100,10)}

# clf=AdaBoostClassifier(n_estimators=70,learning_rate=1.5,random_state=33)
# params={'n_estimators':np.arange(10,200,10)}
# params={'learning_rate': np.arange(0.5, 2.0, 0.1)}

# parm_search(clf,params,X_train,y_train)


# # model training
xgbc=XGBClassifier(n_estimators=85,max_depth=4)
xgbc.fit(X_train,y_train)
y_pro=xgbc.predict_proba(X_test)
y_pro=np.delete(y_pro,-1,axis=1)
y_pred=[]

for i in range(y_pro.shape[0]):
    if y_pro[i][0]>0.5:
        y_pred.append(0)
    else:
        y_pred.append(1)

# print score(y_val,y_pred)
to_submission(y_pred,Id)


# feature: ['stop_cnt':stop_cnt,'start_x':start_x,'xv_end':xv_end,xv_std,xv_skew,ax_std,ax_max,
# x_skew,x_kurt,x_avg,y_std,y_avg,time_std,time_skew,time_avg,time_max,d_skew,d_avg,tan_skew,cos_skew,cos_avg]
# xgbc=XGBClassifier(n_estimators=100,learning_rate=0.1,min_child_weight=1)
# online: 13093(76.31)


# feature: ['tot_time','start_speed','median_speed','end_speed','avg_speed','distance','sum_distance','tan']
# xgb = XGBClassifier(n_estimators=290,learning_rate=0.1)
# offline: 91.01(25368) online: 58.58


#feature: ['start_speed''median_speed','eighty_per_speed','end_speed','tan','max_tan_difference','tot_time','sum_distance']
# XGBClassifier(n_estimators=210,learning_rate=0.21,max_depth=5,min_child_weight=1,gamma=0,subsample=0.8,colsample_bytree=0.8,scale_pos_weight=1)
# offline: 94.259(24217) online: 58.47

# feature: ['end_speed','tan','start_x']
# XGBClassifier(n_estimators=200,learning_rate=0.1,min_child_weight=1)
# offline: 84.02(15670) online: 59.44

# feature: ['start_speed','twenty_per_speed','median_speed','eighty_per_speed','end_speed','avg_speed','distance','tan','max_tan_difference']
# XGBClassifier(n_estimators=100,learning_rate=0.6,min_child_weight=1)
# offline: 90.45(20226) online: 54.18

# feature: ['start_speed','median_speed','end_speed','avg_speed','distance','tan','max_tan_difference']
# XGBClassifier(n_estimators=300,learning_rate=0.8,min_child_weight=1)
# offline: 88.039(22731) online:


# feature: ['tot_time','start_speed','median_speed','end_speed','avg_speed','distance','sum_distance','tan']
# XGBClassifier(n_estimators=220,learning_rate=0.41,min_child_weight=2.3)
# offline: 90.71(25353) online: 67.07