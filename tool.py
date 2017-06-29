# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

def gen_black(data1,data2):
    n=len(data1)
    m=len(data2)
    print n,m
    i=j=0
    cnt=0
    data1=data1['id']
    data2=data2['id']
    res=[]
    while i<n and j<m:
        if data1[i]==data2[j]:
            cnt+=1
            res.append(data1[i])
            i+=1
            j+=1
        else:
            if data1[i]>data2[j]:
                # res.append(data2[j])
                j+=1
            else:
                # res.append(data1[i])
                i+=1
    print cnt
    res_df=pd.DataFrame({'id':res,'label':0})
    res_df.to_csv('/home/frank/data/mouse/black.csv',index=None)

#  扩充训练集
def expand_training():
    data1 = pd.read_csv('/home/frank/data/mouse/13893(76.09: 17452&25020).csv')
    data2 = pd.read_csv('/home/frank/data/mouse/13745(76.14: 17177&25020).csv')
    gen_black(data1,data2)
    train_df = pd.read_csv('/home/frank/data/mouse/train.csv')
    test_df = pd.read_csv('/home/frank/data/mouse/test.csv')
    black_df=pd.read_csv('/home/frank/data/mouse/black.csv')
    #
    add_train_df=pd.merge(test_df,black_df,on='id')
    add_train_df=add_train_df.drop(['id'],axis=1)
    train_df=pd.concat([train_df,add_train_df])
    train_df.to_csv('/home/frank/data/mouse/expand_train.csv',index=None)


expand_training()