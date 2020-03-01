# -*- coding: utf-8 -*-
"""
Created on Tue Jan 10 14:45:29 2017

@author: victor

metric computation

"""
from src.submodular import *
from src.constraints_tools import *

"""
QP Learning functions

p is the power of norm
style is (k) for use kadd or (0) for submodular

"""






"""
LP Learning functions


p is the power of norm
style is (k) for use kadd or (0) for submodular

"""
#
#def ChoqLP(X,Y,style,p,K):
#
#    X = preprocessing.scale(X)
#    m = preprocessing.MinMaxScaler()
#    X = m.fit_transform(X)
#    dim = len(X[0])
#    number_of_constraints = 200
#    max_iter = 30
#    A=GenerateTriplets(Y,number_of_constraints, max_iter)
#    #print(A)
#    V =GenerateConstraints(A,X,p)
#    #print(V)
#    A = GenrateSortedConstraints(V)
#    #print(A)
#    m = 1.0
#    bc = cvx.matrix(m,(number_of_constraints,1))
#    if style == 0 :
#        AZ = submodular(2**dim)
#    else:
#        AZ = k_additivity(2**dim,min(style,dim-1))
#    AZ = cvx.matrix(AZ)
#    bs = cvx.matrix(0.0,(np.shape(AZ)[0],1))
#    #print(AZ)
#    #AZ, bs, bas = convert2kadd(AZ,bs)
#    #AP = cvx.matrix([(-1)*cvx.spmatrix(1.0, range(2**dim), range(2**dim)),cvx.spmatrix(1.0, range(2**dim), range(2**dim))])
#    #bp = cvx.matrix([cvx.matrix(0.0,(2**dim,1)),cvx.matrix(1.0,(2**dim,1))])
#    AP = (-1)*cvx.spmatrix(1.0, range(2**dim), range(2**dim))
#    bp = cvx.matrix(0.0,(2**dim,1))
#
#    c = cvx.matrix(1.0,(2**dim,1))
#    G = cvx.matrix([A,AZ,AP],tc = 'd')
#    h = cvx.matrix([bc,bs,bp])
#    s = cvx.solvers.lp(c,G,h)
#    mu = s['x']
#    print(mu.T)
#    score = ComputeScore(X,Y,K,dim,mu,ChoMetric,p)
#    return score


"""
QP MultiLinear Learning functions

p is the power of norm
style is (k) for use kadd or (0) for submodular

"""


def MLQP(X, Y, style, p, num_constraints):
    X = preprocessing.scale(X)
    m = preprocessing.MinMaxScaler()
    X = m.fit_transform(X)
    dim = len(X[0])
    max_iter = min(len(X) / 2, 200)


    if style == 0:
        AZ = submodular(2 ** dim)
    else:
        AZ = k_additivity(2 ** dim, min(style, dim - 1))
    bs = cvx.matrix(0.0, (AZ.size[0], 1))

    AP = (-1) * cvx.spmatrix(1.0, range(2 ** dim), range(2 ** dim))
    bp = cvx.matrix(0.0, (2 ** dim, 1))

    if(num_constraints<100):
        num_constraints = int(max(num_constraints *AZ.size[0],2))

    A = GenerateTriplets(Y, num_constraints, max_iter)
    V = GenerateConstraints(A, X, p)
    A = GenrateMLConstraints(V)

    a = 0.3
    P = 2 * (1 - a) * cvx.spmatrix(1.0, range(2 ** dim), range(2 ** dim))
    q = cvx.matrix(a, (2 ** dim, 1))
    G = cvx.matrix([A, AZ, AP], tc='d')
    margin = 1.0
    bc = cvx.matrix(margin, (num_constraints, 1))
    h = cvx.matrix([bc, bs, bp])
    s = cvx.solvers.qp(P, q, G, h)
    mu = s['x']
    print(mu.T)

    return mu