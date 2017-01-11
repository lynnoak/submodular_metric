import sys
sys.path.append("./src")

from genutils import *
from base import *
from sublearning import *
import numpy as np
import timeit

from mydatasets import *
from math import log2
import cvxopt as cvx
from sklearn import *
from itertools import *
from constraints_tools import *
from metric_computation import *
from show_tools import *

from sklearn.decomposition import PCA


#def mafonction(x):
#	return x*x*x - x*x - 2
#
#
#def timer(call):
#	start_time = timeit.default_timer()
#	call
#	print(timeit.default_timer() - start_time)
#

#timer(Bisection(1,2,mafonction,100))


#mu = [0,0.18,0.15,0.28,0.23,0.48,0.56,1]
#x = [0.1, 0.2, 0.6]
#print("Input Entry :",x)
#print("Set function : ",mu)
#print(Choquet(x,mu))
#print(ChoquetGradient(x,mu))#


#X = np.random.rand(20,3)
#Y = [1,2,2,1,3,4,3,4,1,3,2,4,1,4,2,3,1,2,3,4]
#print("Label vector : ",Y)
#number_of_constraints = 10
#max_iter = 30
#A=GenerateTriplets(Y,5, max_iter)
#print(A)
#V =GenerateConstraints(A,X)
#print(V)
#

"""
Test for the functions
"""


X,Y = iono_data()
X = preprocessing.scale(X)
m = preprocessing.MinMaxScaler()
X = m.fit_transform(X)  

dim = len(X[0])

number_of_constraints = 100
max_iter = 30
A=GenerateTriplets(Y,number_of_constraints, max_iter)
#print(A)
print("GT is ok")
V =GenerateConstraints(A,X)
#print(V)

A = GenrateSortedConstraints(V)
#print(A)
print("A is ok")

#AZ = convexity(2**dim)    
AZ = k_additivity(2**dim,k=3)
#print(AZ)
print("AZ is ok")

G = cvx.matrix([A,AZ],tc = 'd')


a = 0.3
m = 1.0

P = 2*(1-a)*cvx.spmatrix(1.0, range(2**dim), range(2**dim))
q = cvx.matrix(a,(2**dim,1))
bc = cvx.matrix(m,(number_of_constraints,1))
[t,t1] = AZ.size
bs = cvx.matrix(0.0,(t,1))
h = cvx.matrix([bc,bs])


s = cvx.solvers.qp(P,q,G,h)
mu = s['x']
print(mu.T)

K = 3
score = ComputeScore(X,Y,K,dim,mu,ChoMetric,1)

KNN_score = ComputeKNNScore(X,Y,K,1)


"""
Test for show the result
"""

#
#myLoadData=[balance_data(),seeds_data(),wine_data(),iono_data(),sonar_data()]
#SKNN = []
#stdSKNN = []
#SCho = []
#stdSCho = []
#
#
#
#for i in myLoadData:
#    X,Y = i
#    
#    #reduce the instance 
#    n = max(len(X),300)
#    X,Y = X[0:n],Y[0:n]
#
#       
#    #reduce the dimension
#    PCAK = 20
#    if  (len(X[0])>PCAK ) :
#        pca = PCA(n_components=PCAK)
#        X = pca.fit_transform(X)
#
#
#    X = preprocessing.scale(X)
#    m = preprocessing.MinMaxScaler()
#    X = m.fit_transform(X)
# 
#    dim = len(X[0])
#    number_of_constraints = 100
#    max_iter = 30
#    A=GenerateTriplets(Y,number_of_constraints, max_iter)
#    V =GenerateConstraints(A,X)
#    A = GenrateSortedConstraints(V)
#    AZ = convexity(2**dim)
#    #AZ = k_additivity(2**dim,k=2)
#    G = cvx.matrix([A,AZ],tc = 'd')
#    
#    a = 0.3
#    P = 2*(1-a)*cvx.spmatrix(1.0, range(2**dim), range(2**dim))
#    q = cvx.matrix(a,(2**dim,1))
#    
#    m = 1.0
#    bc = cvx.matrix(m,(number_of_constraints,1))
#    [t,t1] = AZ.size
#    bs = cvx.matrix(0.0,(t,1))
#    h = cvx.matrix([bc,bs])
#    s = cvx.solvers.qp(P,q,G,h)
#    mu = s['x']
#    print(mu.T)
#    
#    K = 3
#    mean,std = ComputeScore(X,Y,K,dim,mu,ChoMetric,1)
#    SCho.append(mean)
#    stdSCho.append(std) 
#    
#    mean,std = ComputeKNNScore(X,Y,K,1)
#    SKNN.append(mean)
#    stdSKNN.append(std)
#    
#ShowBar(SKNN,stdSKNN,SCho,stdSCho)
#
#    
