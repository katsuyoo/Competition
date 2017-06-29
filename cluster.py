# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

test_df=pd.read_csv('/home/frank/data/mouse/test.csv')

X=test_df.drop(['id'],axis=1).values
kmeans=KMeans(n_clusters=2).fit(X)
lables=kmeans.labels_
print lables.sum()
