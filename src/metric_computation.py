# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 14:45:29 2017

@author: victor

metric computation

"""
import sys
sys.path.append("..")
from base import *
from sublearning import *
import numpy as np
from sklearn import *
from sklearn.neighbors import *
from sklearn.metrics import *



def ChoMetric(X1,X2,**kwargs):
    """
    define the metric with Choquet extension 
    Input: pair of object and the configuration of
        "dim" number of dimension
         "M" set function
         "p" the power of the norm
    Output: the choquet metric
        
    """
    X = abs(X1-X2)
    if len(X)!= kwargs["dim"]:
        M = np.arange(0,pow(2,len(X)))
        M = [bitCount(i) for i in M]
        M = np.array(M,dtype= np.double)
        return Choquet(X,M)
    else:
        X = [abs(i)**kwargs["p"] for i in X]
        X = np.array(X,dtype= np.double) 
        r = Choquet(X,kwargs["M"])
        return r**(1/kwargs["p"])


def ColKNNScoreSM(X,Y,K,M,pp,metric=ChoMetric,scoring = 'acc', title=""):
    """
    compute the cross validation score of the metric
    Input: "X,Y" the dataset
        "K" K of KNN alg
        "M" set function
         "metric" metric computation function
    Output: the mean and std of score 
        
    """
    S = {}
    dim = len(X[0])

    X = preprocessing.scale(X)
    m = preprocessing.MinMaxScaler()
    X = m.fit_transform(X)

    ditscoring = {'acc':'accuracy',
                  'pre':metrics.make_scorer(metrics.precision_score,average = 'weighted'),
                  'rec': metrics.make_scorer(metrics.recall_score, average='weighted'),
                  'f_1': metrics.make_scorer(metrics.f1_score, average='weighted')}

    if (scoring == 'test'):
        scoring = ['acc','f_1','pre','rec']

    if not isinstance(scoring,list):
         scoring = [scoring]

    for s in scoring:
         S_mea = []
         S_std = []
         for i in range(5):
             myKNN = KNeighborsClassifier(n_neighbors=K,
                                          metric=metric, metric_params={"M": M, "dim": dim,"p":pp})
             myKNN.fit(X, Y)
             kf = model_selection.StratifiedKFold(n_splits=3, shuffle=True)
             score_KNN = model_selection.cross_val_score(myKNN, X, Y, cv=kf,
                                                         scoring=ditscoring.get(s, 'accuracy'))
             S_mea.append(score_KNN.mean())
             S_std.append(score_KNN.std() * 2)
         S_mea = np.mean(S_mea)
         S_std = np.mean(S_std)
         print(title + " " + s + " : %0.4f(+/-)%0.4f " % (S_mea, S_std))
         S[s] = (S_mea, S_std)

    return S



def ColKNNScore(X,Y,K,p,scoring = 'acc',title = ""):
    """
    compute the cross validation score of the KNN as Control
    Input: "X,Y" the dataset
        "K" K of KNN alg
         "p" the power of the norm
    Output: the mean and std of KNNscore 
        
    """

    S = {}

    ditscoring = {'acc':'accuracy',
                  'pre':metrics.make_scorer(metrics.precision_score,average = 'weighted'),
                  'rec': metrics.make_scorer(metrics.recall_score, average='weighted'),
                  'f_1': metrics.make_scorer(metrics.f1_score, average='weighted')}

    if (scoring == 'test'):
        scoring = ['acc','f_1','pre','rec']

    if not isinstance(scoring,list):
        scoring = [scoring]

    for s in scoring:
        S_mea = []
        S_std = []
        for i in range(5):
            KNN = KNeighborsClassifier(n_neighbors=K, p=p)
            KNN.fit(X, Y)
            kf = model_selection.StratifiedKFold(n_splits=3, shuffle=True)
            score_KNN = model_selection.cross_val_score(KNN, X, Y, cv=kf,
                                                        scoring=ditscoring.get(s, 'accuracy'))
            S_mea.append(score_KNN.mean())
            S_std.append(score_KNN.std()*2)
        S_mea = np.mean(S_mea)
        S_std = np.mean(S_std)
        print(title + " " + s + " : %0.4f(+/-)%0.4f " % (S_mea,S_std))
        S[s] =  (S_mea,S_std)
    return S



def Mul_linear(X, M):
    dim = len(X)
    ml = 0
    for i in range(2**dim):
        bii = [1 if i & (1 << (dim-1-n)) else 0 for n in range(dim)]
        bii = np.array(bii)
        tt = bii*X+(1-bii)*(1-X)
        ml += M[i]*np.prod(tt)

    return ml

def MLMetric(X1, X2, **kwargs):
    """
       define the metric with Choquet extension
       Input: pair of object and the configuration of
           "dim" number of dimension
            "M" set function
            "p" the power of the norm
       Output: the choquet metric

   """

    X = abs(X1 - X2)
    if len(X) != kwargs["dim"]:
        M = np.arange(0, pow(2, len(X)))
        M = [bitCount(i) for i in M]
        M = np.array(M, dtype=np.double)
        return Mul_linear(X, M)
    else:
        X = [abs(i) ** kwargs["p"] for i in X]
        X = np.array(X, dtype=np.double)
        r = Mul_linear(X, kwargs["M"])
        return r ** (1 / kwargs["p"])