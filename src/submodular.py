# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 14:45:29 2017
@author: victor
Submodular Metric Learning
"""

from src.myMLtools import *
from src.constraints_tools import *

class Submodular(object):
    """
    Learning the submodular metric and return the score of the KNN
    :param X: feature of dataset
    :param Y: target label
    :param style: 0: submodular, int>2: k_additivity
    :param num_constraints: Number of selection constraints from the label
    """

    def __init__(self,X = X, y=y,style = 0, P_power =2, num_constraints =100):
        X = normalize(X, axis=0)
        self.X = X
        self.y = y
        self.style = style
        self.P_power = P_power
        self.num_constraints = num_constraints

        X = preprocessing.scale(self.X)
        m = preprocessing.MinMaxScaler()
        X = m.fit_transform(X)
        dim = len(X[0])
        max_iter = min(len(X) / 2, 200)

        time_start =time.clock()

        if style == 0:
            AZ = submodular(2 ** dim)
        else:
            AZ = k_additivity(2 ** dim, min(style, dim - 1))
        bs = cvx.matrix(0.0, (AZ.size[0], 1))

        AP = (-1) * cvx.spmatrix(1.0, range(2 ** dim), range(2 ** dim))
        bp = cvx.matrix(0.0, (2 ** dim, 1))

        if (num_constraints < 100):
            num_constraints = int(max(num_constraints * AZ.size[0], 2))

        A = GenerateTriplets(self.y, num_constraints, max_iter)
        V = GenerateConstraints(A, X, self.P_power)
        A = GenrateLovaszConstraints(V)

        a = 0.3
        P = 2 * (1 - a) * cvx.spmatrix(1.0, range(2 ** dim), range(2 ** dim))
        q = cvx.matrix(a, (2 ** dim, 1))
        G = cvx.matrix([A, AZ, AP], tc='d')
        margin = 1.0
        bc = cvx.matrix(margin, (num_constraints, 1))
        h = cvx.matrix([bc, bs, bp])
        s = cvx.solvers.qp(P, q, G, h)

        time_end =time.clock()
        self.SMtime =str(time_start-time_end)
        self.SMmetric = s['x']

    def GetPara(self):
        print(self.style)
        print(self.SMmetric)

    def ShowTime(self):
        if self.style == 0:
            print("Learning Time for Submodular is " + self.SMtime)
            print('\n')
        else:
            print("Learning Time for "+ str(self.style)+"-additivity is " + self.SMtime)
            print('\n')

    def ChoMetric(self, X1, X2):
        """
        define the metric with Choquet extension
        Input: pair of object and the configuration of
        Output: the choquet metric

        """
        X = abs(X1 - X2)
        X = [abs(i) ** self.P_power for i in X]
        X = np.array(X, dtype=np.double)
        r = Choquet(X, self.SMmetric)
        r = abs(r) ** (1 / self.P_power)
        if type(r) == float:
            return r
        else:
            print(X)
            print(r)
            return 0

    def ColKNNScore(self,n_neighbors = 5, scoring = ['acc']):
        if self.style ==0:
            title ="Submodular"
        else:
            title = str(self.style)+"-additivity"

        metric = 'pyfunc'
        metric_params = {"func":self.ChoMetric}
        CKS = KNN_score(n_neighbors =n_neighbors,P_power =self.P_power, scoring=scoring,metric=metric,metric_params=metric_params)
        return CKS.ColKNNScore(self.X,self.y,title=title)
