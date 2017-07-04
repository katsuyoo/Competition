# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import scipy as sp
import operator
from scipy import stats as sts


def read_test():
    fo=open('/home/frank/data/mouse/dsjtzs_txfz_test1.txt')
    test_data=fo.readlines()
    id=[]
    data=[]
    p_data=[]
    for line in test_data:
        line=line.split(' ')
        id.append(line[0])
        pos=line[-2].split(',')
        pos[0]=float(pos[0])
        pos[1]=float(pos[1])
        p_data.append(tuple(pos))
        #print line
        node_list=line[1].split(';')[:-1]
        #print node_list
        sample=[]
        for node in node_list:
            info=node.split(',')
            for i in range(3):
                info[i]=float(info[i])
            sample.append(tuple(info))
        # sample.sort(key=operator.itemgetter(2))
        data.append(tuple(sample))
    data=np.array(data)
    return id,data,p_data

id,data,p_data=read_test()
n=len(data)
print n
cnt=0
for i in range(n):
    flag=1
    m=len(data[i])
    for j in range(m-1):
        if data[i][j+1][-1]<=data[i][j][-1]:
            flag=0
            break
    if flag==1:
        cnt+=1

print cnt