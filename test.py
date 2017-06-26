# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

def calc(n,m):
    p=n*1.0/m
    r=m/20000.0
    print 5*p*r/(2*p+3*r)*100.0


def count(data1,data2):
    n=len(data1)
    m=len(data2)
    print n,m
    i=j=0
    cnt=0
    data1=data1['id']
    data2=data2['id']
    while i<n and j<m:
        if data1[i]==data2[j]:
            cnt+=1
            i+=1
            j+=1
        else:
            if data1[i]>data2[j]:
                j+=1
            else:
                i+=1
    print cnt


# calc(12941,25350)
# calc(11100,20569)
# calc(12941,24217)
# data1=pd.read_csv('/home/frank/data/mouse/submission.csv')
# data2=pd.read_csv('/home/frank/data/mouse/submission4.csv')
# count(data1,data2)

# train_df=pd.read_csv('/home/frank/data/mouse/train.csv')
# X=train_df.drop(['label'],axis=1)
# y=train_df['label']
# X=X.values
# y=y.values
#
#
# kf=KFold(n_splits=5,shuffle=True)
# #
# for i,(train_index,test_index) in enumerate(kf.split(X,y)):
#     y_train = y[train_index]
#     cnt=0
#     for j in range(len(y_train)):
#         if y_train[j]==0:
#             cnt+=1
#     print cnt
    # break


# base_model = [
#     ['lsvc', 1],
#     ['xgbc', 2],
#     ['rfc', 3],
#     ['bgc', 4]
# ]

# for i,bm in enumerate(base_model):
#     print i,bm[1]
# l=[
#     [1,2],
#     [3,4],
#     [5,6]
# ]
# a=np.array(l)
# print a.shape
# print a.mean(axis=1)

