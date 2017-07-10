import sys
sys.path.append("./src")

from genutils import *
from base import *
from sublearning import *
import numpy as np
import timeit

from mydatasets import *
from time import clock
from math import log2
import cvxopt as cvx
from sklearn import *
from itertools import *
from constraints_tools import *
from metric_computation import *

from sklearn.decomposition import PCA
from metric_learn import LMNN,ITML_Supervised,LSML_Supervised



"""
QP Learning functions

p is the power of norm
style is (k) for use kadd or (0) for submodular

"""

def ChoqQP(X,Y,style,p,K):
    
    X = preprocessing.scale(X)
    m = preprocessing.MinMaxScaler()
    X = m.fit_transform(X)  		
    dim = len(X[0])		
    number_of_constraints = 200
    max_iter = min(len(X)/2,200)
    A=GenerateTriplets(Y,number_of_constraints, max_iter)
    #print(A)
    V =GenerateConstraints(A,X,p)
    #print(V)		
    A = GenrateSortedConstraints(V)
    #print(A)
    m = 1.0
    bc = cvx.matrix(m,(number_of_constraints,1))	
    if style == 0 :
        AZ = submodular(2**dim)
    else:
        AZ = k_additivity(2**dim,min(style,dim-1))
    AZ = cvx.matrix(AZ)
    bs = cvx.matrix(0.0,(np.shape(AZ)[0],1))
    #print(AZ)	
    #AZ, bs, bas = convert2kadd(AZ,bs)		
    #AP = cvx.matrix([(-1)*cvx.spmatrix(1.0, range(2**dim), range(2**dim)),cvx.spmatrix(1.0, range(2**dim), range(2**dim))])
    #bp = cvx.matrix([cvx.matrix(0.0,(2**dim,1)),cvx.matrix(1.0,(2**dim,1))])		
    AP = (-1)*cvx.spmatrix(1.0, range(2**dim), range(2**dim))
    bp = cvx.matrix(0.0,(2**dim,1))
      		
    a = 0.3
    P = 2*(1-a)*cvx.spmatrix(1.0, range(2**dim), range(2**dim))
    q = cvx.matrix(a,(2**dim,1))
    G = cvx.matrix([A,AZ,AP],tc = 'd')
    h = cvx.matrix([bc,bs,bp])
    s = cvx.solvers.qp(P,q,G,h)
    mu = s['x']
    print(mu.T)		
    score = ComputeScore(X,Y,K,dim,mu,ChoMetric,p)
    return score
       	
"""
test
"""


K = 5#K for knn
X,Y = mnist_data()

#reduce the dimension
PCAK = 8
if  (len(X[0])>PCAK ) :
    pca = PCA(n_components=PCAK)
    X = pca.fit_transform(X)	


s0 = clock()
Mah_score = ComputeKNNScore(X,Y,K,1)	
s1 = clock()
Mah_t = s1-s0
print('OrgKNN p1 time is ',s1-s0)
s0 = s1

Eud_score = ComputeKNNScore(X,Y,K,2)	
s1 = clock()
Eud_t = s1-s0
print('OrgKNN p2 time is ',s1-s0)
s0 = s1

lmnn = LMNN(k=5, learn_rate=1e-6)
lmnn.fit(X,Y)
XL = lmnn.transform(X)
S_LMNN = ComputeKNNScore(XL,Y,K,2)
s1 = clock()
LMNN_t = s1-s0
print('lmnnKNN time is ',s1-s0)
s0 =s1

itml = ITML_Supervised(num_constraints=200)
itml.fit(X,Y)
XI = itml.transform(X)
S_ITML = ComputeKNNScore(XI,Y,K,2)
s1 = clock()
ITML_t = s1-s0
print('itmlKNN time is ',s1-s0)
s0 =s1

lsml = LSML_Supervised(num_constraints=200)
lsml.fit(X,Y)
XL = lsml.transform(X)
S_LSML = ComputeKNNScore(XL,Y,K,2)
s1 = clock()
LSML_t = s1-s0
print('lsmlKNN time is ',s1-s0)
s0 =s1


style = 0

Chq_1_score = ChoqQP(X,Y,style,1,K)
s1 = clock()
Chp_1_t = s1-s0
print('ChqKNN p1 time is ',s1-s0)
s0 =s1

s0 = s1
Chq_2_score = ChoqQP(X,Y,style,2,K)
s1 = clock()
Chp_2_t = s1-s0
print('ChqKNN p2 time is ',s1-s0)
s0 =s1


#
#Chq_k_score = []
#Chq_k_t = []
#for style in range(1,len(X[0])+1):
#    Chq_k_score.append( ChoqQP(X,Y,style,2,K))
#    s1 = clock()
#    Chq_k_t.append(s1-s0)
#    print('kadd',style,'time is',s1-s0)
#    s0 = s1


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

#
#K = 5#K for knn
#X,Y = iono_data()
##reduce the dimension
#PCAK = 8
#if  (len(X[0])>PCAK ) :
#    pca = PCA(n_components=PCAK)
#    X = pca.fit_transform(X)	
#
#
#s0 = clock()
#
##Mah_score = ComputeKNNScore(X,Y,K,1)	
##s1 = clock()
##print('OrgKNN p1 time is ',s1-s0)
##s0 = s1
##
##Eud_score = ComputeKNNScore(X,Y,K,2)	
##s1 = clock()
##print('OrgKNN p2 time is ',s1-s0)
##s0 = s1
##
#style = 0
#
#Chq_1_score = ChoqLP(X,Y,style,1,K)
#s1 = clock()
#print('ChqKNN p1 time is ',s1-s0)
#s0 =s1
#
#s0 = s1
#Chq_1_score = ChoqLP(X,Y,style,2,K)
#s1 = clock()
#print('ChqKNN p2 time is ',s1-s0)
#s0 =s1

