# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 14:45:29 2017

@author: victor

metric computation

"""

import numpy as np
from sklearn import *
from sklearn.neighbors import *
from sklearn.metrics import *

from metric_learn import *
import time

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class KNN_score(object):
    """
    Compute the cross validation score of the KNN
    :param n_neighbors: K of KNN algorithm
    :param scoring: the configuration of the output,
                {'acc': accuracy rate,
                  'pre': precision score,
                  'rec': recall score,
                  'f_1': f1 score, }
    """
    ditscoring = {'acc': 'accuracy',
              'pre': make_scorer(precision_score, average='weighted'),
              'rec': make_scorer(recall_score, average='weighted'),
              'f_1': make_scorer(f1_score, average='weighted')}

    def __init__(self,n_neighbors = 5 ,P_power = 2, scoring = ['acc'], metric = 'minkowski', metric_params={}):
        self.n_neighbors = n_neighbors
        self.P_power = P_power
        test = ['acc', 'f_1', 'pre', 'rec']
        if scoring =='test' or scoring==['test']:
            scoring =test
        if not isinstance(scoring, list):
            scoring = [scoring]
        if scoring==[] or not set(test) >= set(scoring):
            scoring = ['acc']
        self.scoring =scoring
        self.metric = metric
        self.metric_params = metric_params

    def GetPara(self):
        print(self.__dict__)
        return self.__dict__

    def ColKNNScore(self, X, y, title=""):
        """
        Compute the cross validation score of the KNN

        :param X: Data feature
        :param y: Label
        :param title: title information
        :return: S = (S_mea, S_std):the mean and std of KNNscore
        """
        S = {}
        for s in self.scoring:
            S_mea = []
            S_std = []
            for i in range(5):
                KNN = KNeighborsClassifier(n_neighbors=self.n_neighbors, p=self.P_power,algorithm='ball_tree',metric=self.metric,metric_params=self.metric_params)
                KNN.fit(X, y)
                kf = model_selection.StratifiedKFold(n_splits=3, shuffle=True)
                score_KNN = model_selection.cross_val_score(KNN, X, y, cv=kf,
                                                            scoring=self.ditscoring.get(s, 'accuracy'))
                S_mea.append(score_KNN.mean())
                S_std.append(score_KNN.std() * 2)
            S_mea = np.mean(S_mea)
            S_std = np.mean(S_std)
            print(title + " " + s + " : %0.4f(+/-)%0.4f " % (S_mea, S_std))
            S[s] = (S_mea, S_std)
        return S

data= datasets.load_iris()
X = data['data']
y = data['target']

def error_metric(**kwargs):
    return 0

class NormalML(KNN_score):
    """
    Learning the submodular metric and return the score of the KNN
    :param X: feature of dataset
    :param Y: target label
    :param alg: the list of name of Metric Learning algorithm
    :param num_constraints: Number of selection constraints from the label
    """
    MLerror = {}
    X_ML = {}
    metricLearnt = {}
    MLtime ={}
    def __init__(self,alg=['NCA'], X=X, y=y, num_constraints=100,n_neighbors = 5 ,P_power = 2, scoring = ['acc'], metric = 'minkowski', metric_params={}):
        super(NormalML,self).__init__(n_neighbors,P_power,scoring, metric, metric_params)
        X = normalize(X,axis=0)
        self.X = X
        self.y = y
        test = ['NCA','LSML','LFDA']
        allalg = ['NCA','LSML','LFDA','ITML','LMNN']
        if alg =='test' or alg ==['test']:
            alg = test
        if alg == 'all' or alg == ['all']:
            alg = allalg
        if not isinstance(alg, list):
            alg = [alg]
        if alg==[] or not set(allalg) >= set(alg):
            alg = ['NCA']
        self.alg = alg
        self.num_constraints = num_constraints

        ditalg = {'NCA': NCA(),
                  'LSML': LSML_Supervised(num_constraints=self.num_constraints),
                  'LFDA': LFDA(),
                  'ITML': ITML_Supervised(num_constraints=self.num_constraints),
                  'LMNN': LMNN()}
        for i in self.alg:
            algmodel = ditalg[i]
            time_start = time.clock()
            try:
                try:
                    algmodel.fit(X, y)
                except:
                    tX = X + 10 ** -4
                    algmodel.fit(tX, y)
                self.MLerror[i] = 0
                self.X_ML[i] = algmodel.transform(X)
                self.metricLearnt[i] =algmodel.get_metric()
            except:
                print(i+" get error!\n")
                self.MLerror[i] = 1
                self.X_ML[i] = X
                self.metricLearnt[i] = error_metric

            time_end = time.clock()
            self.MLtime[i] = str(time_start - time_end)

    def GetPara(self):
        print(self.alg)
        print(self.MLerror)
        return self.__dict__

    def ShowTime(self):
        for i in self.alg:
            print("Learning Time for "+i+" is " + self.MLtime[i])
            print('\n')

    def ColKNNScore(self):
        S = {}
        S['Euc.'] = super().ColKNNScore(self.X,self.y,title='Euc. ')
        pca = decomposition.PCA(n_components=0.9)
        X_pca = pca.fit_transform(self.X,self.y)
        if len(X_pca[0]) < 3:
            pca = decomposition.PCA(n_components=3)
            X_pca = pca.fit_transform(self.X,self.y)

        S['PCA'] = super().ColKNNScore(X_pca,self.y,title='PCA ')
        for i in self.alg:
            if self.MLerror[i] == 0:
                S[i] = super().ColKNNScore(self.X_ML[i],self.y,title=i)
            else:
                S[i] = (0,0)
        return S

def my3Dplot(X, y, title=" "):
    """
    Show the 3D vesion of the distance in the latent space

    :param X: Data feature
    :param y: Label
    :param title: title information
    :return: the figure
    """

    fig = plt.figure()
    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = le.transform(y)
    color = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    color = [color[int(i) % 7] for i in y]
    PCAK = 3
    if (len(X[0]) > PCAK):
        pca = PCA(n_components=PCAK)
        X = pca.fit_transform(X)
        X = normalize(X,axis=0)

    ax = plt.subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color)
    ax.set_title(title)

    plt.savefig(title)
    plt.show()
