# -*- coding: utf-8 -*-
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from xgboost.sklearn import XGBClassifier
from sklearn.svm import LinearSVC,SVC
from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression



class Stack(object):
    def __init__(self):
        self.random_rate=33
        clf1=SVC(C=1.0,random_state=33)
        clf2=XGBClassifier(n_estimators=220,learning_rate=0.2,min_child_weight=2.3)
        clf3=RandomForestClassifier(n_estimators=80,random_state=330,n_jobs=-1)
        clf4=BaggingClassifier(n_estimators=40,random_state=101)
        clf5=AdaBoostClassifier(n_estimators=70,learning_rate=1.5,random_state=33)
        clf6=GradientBoostingClassifier(n_estimators=250,learning_rate=0.23,random_state=33)

        clf7=XGBClassifier(n_estimators=100,learning_rate=0.12,min_child_weight=1)


        base_model=[
            ['svc',clf1],
            ['xgbc',clf2],
            ['rfc',clf3],
            ['bgc',clf4],
            ['adbc',clf5],
            ['gdbc',clf6]
        ]

        self.base_models=base_model
        self.XGB=clf7

    def stacking(self,X,y,test):
        models=self.base_models
        kf=KFold(n_splits=5,shuffle=True,random_state=4670)
        folds=list(kf.split(X,y))
        s_train=np.zeros((X.shape[0],len(models)))
        # s_test=np.zeros((test.shape[0],len(models)))

        for i,bm in enumerate(models):
            clf=bm[1]
            # s_test_i = np.zeros((test.shape[0], len(folds)))
            for j,(train_idx,test_idx) in enumerate(kf.split(X,y)):
                X_train=X[train_idx]
                y_train=y[train_idx]
                X_test=X[test_idx]
                clf.fit(X_train,y_train)
                y_pred=clf.predict(X_test)[:]
                s_train[test_idx,i]=y_pred     #第i个模型的预测值
                # s_test_i[:,j]=clf.predict(test)[:]
            # s_test[:,i]=s_test_i.mean(1)
        print s_train.shape
        # print s_test.shape

        y_new=s_train.mean(axis=1)
        for i in range(len(y_new)):
            if y_new[i]>0.5:
                y_new[i]=1
            else:
                y_new[i]=0
        print y_new.mean()

        self.XGB.fit(X,y_new)
        yp=self.XGB.predict(test)[:]
        print yp.mean()
        return yp

        pass

