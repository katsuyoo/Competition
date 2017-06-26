# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from stacking import Stack


def standard_data(X):
    X_scaler=StandardScaler()
    X=X_scaler.fit_transform(X)
    return X

def to_submission(y_pred,id):
    n=len(y_pred)
    # print n
    res=[]
    for i in range(n):
        if y_pred[i]==0:
            res.append(id[i])

    print len(res)
    res_df=pd.DataFrame({'id':res})
    res_df.to_csv('/home/frank/data/mouse/submission.csv',index=None)

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

train_df=pd.read_csv('/home/frank/data/mouse/expand_train.csv')
test_df=pd.read_csv('/home/frank/data/mouse/test.csv')

# print train_df.describe()

X=train_df.drop(['label'],axis=1)
y_train=train_df['label']
X_test=test_df.drop(['id'],axis=1)
Id=test_df['id']
X_train=X.values
X_test=X_test.values
# X_train=standard_data(X)
# X_test=standard_data(X_test)

# print X_train.shape[0]

ms=Stack()
y_pred=ms.stacking(X_train,y_train,X_test)
to_submission(y_pred,Id)


# stacking
# feature: ['tot_time','start_speed','median_speed','end_speed','avg_speed','distance','sum_distance','tan']
# base_model:
# clf1 = LinearSVC(C=0.1, random_state=self.random_rate)
# clf2 = XGBClassifier(n_estimators=220, learning_rate=0.41, min_child_weight=2.3)
# clf3 = RandomForestClassifier(n_jobs=-1)
# clf4 = BaggingClassifier(n_jobs=-1)
# clf5 = AdaBoostClassifier()
# two_model:
# clf7=XGBClassifier(n_estimators=210,learning_rate=0.21)
# online: 66.76(20569)

# feature: ['tot_time','start_speed','median_speed','end_speed','avg_speed','distance','sum_distance','tan']
# base_model:
# clf1 = LinearSVC(C=80, random_state=self.random_rate)
# clf2 = XGBClassifier(n_estimators=220, learning_rate=0.41, min_child_weight=2.3)
# clf3 = RandomForestClassifier(n_estimators=70, random_state=8650, n_jobs=-1)
# clf4 = BaggingClassifier(n_estimators=10, random_state=33)
# clf5 = AdaBoostClassifier(learning_rate=0.13, random_state=33)
# clf6 = GradientBoostingClassifier(n_estimators=150, learning_rate=0.11)
# two_model:
# clf7=XGBClassifier(n_estimators=390,learning_rate=0.15)
# online: 68.27(23737)