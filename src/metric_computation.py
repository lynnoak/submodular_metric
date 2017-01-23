# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 14:45:29 2017

@author: victor

metric computation

"""
import sys
sys.path.append("..")
from genutils import *
from base import *
from sublearning import *
import numpy as np
from sklearn import *
from sklearn.neighbors import *
import timeit
import cvxopt as cvx
from itertools import *


def ChoMetric(X1,X2,**kwargs):
    """
    define the metric with Choquet extension 
    Input: pair of object and the configuration of
        "dim" number of dimension
         "v" set function
         "pnorm" the power of the norm
    Output: the choquet metric
        
    """
    X = abs(X1-X2)
    if len(X)!= kwargs["dim"]:
        v = np.arange(0,pow(2,len(X)))
        v = [bitCount(i) for i in v]
        v = np.array(v,dtype= np.double)
        return Choquet(X,v)
    else:
        X = [i**kwargs["pnorm"] for i in X]
        X = np.array(X,dtype= np.double) 
        r = Choquet(X,kwargs["v"])
        return abs(r)**(1/kwargs["pnorm"])


#def MultiMetric(X1,X2,**kwargs):
#    """
#    define the metric with Multilinear extension
#    """



def ComputeScore(X,Y,K,dim,v,metric,pnorm): 
    """
    compute the cross validation score of the metric
    Input: "X,Y" the dataset
        "K" K of KNN alg
        "dim" number of dimension
         "v" set function
         "metric" metric computation function
         "pnorm" the power of the norm
    Output: the mean and std of score 
        
    """
       
    #KNN with the metric 
    myKNN = KNeighborsClassifier(n_neighbors=K, 
                                 metric=metric,metric_params={"dim":dim,"v":v,"pnorm":pnorm})
    myKNN.fit(X,Y)
    #test for predict
    print(myKNN.predict_proba(X[1]))
    #cross validation
    score_my = cross_validation.cross_val_score(myKNN,X,Y,cv=5)
    print(score_my)
    print("Accuracy of myKNN: %0.4f (+/- %0.4f)" % (score_my.mean(), score_my.std()))  
    return score_my.mean(),score_my.std()

def ComputeKNNScore(X,Y,K,pnorm):
    """
    compute the cross validation score of the KNN as Control
    Input: "X,Y" the dataset
        "K" K of KNN alg
         "pnorm" the power of the norm
    Output: the mean and std of KNNscore 
        
    """

    KNN = KNeighborsClassifier(n_neighbors=K,p=pnorm)
    KNN.fit(X,Y)
    print(KNN.predict_proba(X[1]))
    score_KNN = cross_validation.cross_val_score(KNN,X,Y,cv=5)
    print(score_KNN)
    print("Accuracy of KNN: %0.4f (+/- %0.4f)" % (score_KNN.mean(), score_KNN.std()))  
    return score_KNN.mean(),score_KNN.std()