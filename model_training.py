# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import scipy as sp

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from xgboost.sklearn import XGBClassifier
from sklearn.svm import LinearSVC,SVC
from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier



import matplotlib.pyplot as plt
import seaborn as sns


def describe(train_df):
    g = sns.FacetGrid(train_df, col='label')
    g.map(plt.hist, 'tan')
    sns.plt.show()


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
    xgb = XGBClassifier(n_estimators=220,learning_rate=0.41,min_child_weight=2.3)
    xgb.fit(X,y)
    sc=make_scorer(score)
    print cross_val_score(xgb,X,y,scoring=sc,cv=4).mean()
    print xgb.feature_importances_

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
    # print n
    res=[]
    for i in range(n):
        if y_pred[i]==0:
            res.append(id[i])

    print len(res)
    # res_df=pd.DataFrame({'id':res})
    # res_df.to_csv('/home/frank/data/mouse/submission.csv',index=None)



train_df=pd.read_csv('/home/frank/data/mouse/train.csv')
test_df=pd.read_csv('/home/frank/data/mouse/test.csv')
# print test_df.info()

# describe(train_df)
# print train_df['tan'].value_counts()

# print train_df[train_df['label']==1].describe()
# print '-'*100
# print train_df[train_df['label']==0].describe()

X=train_df.drop(['label'],axis=1)
y_train=train_df['label']
X_test=test_df.drop(['id'],axis=1)
Id=test_df['id']

# X_train=X
X_train=standard_data(X)
X_test=standard_data(X_test)

# cv(X_train,y_train)

# clf=XGBClassifier(n_estimators=390,learning_rate=0.15)
# params={'n_estimators':np.arange(100,400,10)}
# params={'learning_rate': np.arange(0.1, 0.3, 0.01)}

# clf=LinearSVC(C=80,random_state=33)
# params={"C":np.arange(10,100,10)}

# clf=RandomForestClassifier(n_estimators=70,random_state=8650)
# params={'n_estimators':np.arange(10,100,10)}

# clf=GradientBoostingClassifier(n_estimators=150,learning_rate=0.11)
# params={'n_estimators':np.arange(100,400,10)}
# params={'learning_rate': np.arange(0.1, 0.3, 0.01)}

# clf=BaggingClassifier(n_estimators=10,random_state=33)
# params={'n_estimators':np.arange(10,100,10)}

clf=AdaBoostClassifier(learning_rate=0.13,random_state=33)
# params={'n_estimators':np.arange(50,200,10)}
params={'learning_rate': np.arange(0.01, 0.15, 0.01)}

parm_search(clf,params,X_train,y_train)


# model training
# xgb=XGBClassifier(n_estimators=390,learning_rate=0.15)
# xgb.fit(X_train,y_train)
# y_pred=xgb.predict(X_test)
# to_submission(y_pred,Id)


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