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
from show_tools import *

from sklearn.decomposition import PCA
from metric_learn import LMNN,ITML_Supervised


#def mafonction(x):
#	return x*x*x - x*x - 2

#
#def timer(call):
#	start_time = timeit.default_timer()
#	call
#	print(timeit.default_timer() - start_time)


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
    max_iter = 30
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
X,Y = iono_data()

#reduce the dimension
PCAK = 8
if  (len(X[0])>PCAK ) :
    pca = PCA(n_components=PCAK)
    X = pca.fit_transform(X)	


s0 = clock()

Mah_score = ComputeKNNScore(X,Y,K,1)	
s1 = clock()
print('OrgKNN p1 time is ',s1-s0)
s0 = s1

Eud_score = ComputeKNNScore(X,Y,K,2)	
s1 = clock()
print('OrgKNN p2 time is ',s1-s0)
s0 = s1

style = 0

Chq_1_score = ChoqQP(X,Y,style,1,K)
s1 = clock()
print('ChqKNN p1 time is ',s1-s0)
s0 =s1

s0 = s1
Chq_1_score = ChoqQP(X,Y,style,2,K)
s1 = clock()
print('ChqKNN p2 time is ',s1-s0)
s0 =s1

lmnn = LMNN(k=5, learn_rate=1e-6)
lmnn.fit(X,Y)
XL = lmnn.transform(X)
S_LMNN = ComputeKNNScore(XL,Y,K,2)
s1 = clock()
print('lmnnKNN time is ',s1-s0)
s0 =s1

itml = ITML_Supervised(num_constraints=200)
itml.fit(X,Y)
XI = itml.transform(X)
S_ITML = ComputeKNNScore(XI,Y,K,2)
s1 = clock()
print('itmlKNN time is ',s1-s0)
s0 =s1

#for style in range(1,11,3):
#    Chq_k_score = ChoqQP(X,Y,style,2,K)
#    s1 = clock()
#    print('kadd',style,'time is',s1-s0)
#    s0 = s1
#
#
#style = floor(len(X[0])/2)
#Chq_k_score = ChoqQP(X,Y,style,2,K)
#s1 = clock()
#print('kadd',style,'time is',s1-s0)
#s0 = s1
#


"""
Test for show the result
"""
#
#def main_show (p = 1,style = 0,PCAK = 8):
#    myLoadData=[glass_data(),iono_data(),sonar_data(),digits_data()]
#    OrgKNN = []
#    stdOrgKNN = []
#    Choq = []
#    stdChoq = []
#    S_LMNN = []
#    stdS_LMNN = []  
#    K = 5
#    
#
#    for i in myLoadData:
#        X,Y = i
#           
#        #reduce the dimension
#        if  (len(X[0])>PCAK ) :
#            pca = PCA(n_components=PCAK)
#            X = pca.fit_transform(X)    
#    
#
#        mean,std = ChoqQP(X,Y,style,p,K)
#        S_LMNN.append(mean)
#        stdS_LMNN.append(std)
#
#        if style == 0 :
#            title = 'test for submodular,p = '+str(p)
#        else:
#            title = 'test for kadd-'+str(style)+',p ='+str(p)
#            AZ = k_additivity(2**dim,min(style,dim-1))
#        ShowBar(OrgKNN,stdOrgKNN,Choq,stdChoq,S_LMNN,stdS_LMNN,title = title)
#        
#        print("mean of OrgKNN",np.mean(OrgKNN))
#        print("mean of Choq",np.mean(Choq))
#        print("mean of S_LMNN",np.mean(S_LMNN)) 
#
#    return OrgKNN,stdOrgKNN,Choq,stdChoq,S_LMNN,stdS_LMNN
#
#
#result11 = main_show(1,0)
#result21 = main_show(2,0)
#result12 = main_show(1,3)
#result22 = main_show(2,3)

"""
LP Learning functions


p is the power of norm
style is (k) for use kadd or (0) for submodular

"""

def ChoqLP(X,Y,style,p,K):
    
    X = preprocessing.scale(X)
    m = preprocessing.MinMaxScaler()
    X = m.fit_transform(X)  		
    dim = len(X[0])		
    number_of_constraints = 200
    max_iter = 30
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
      		
    c = cvx.matrix(1.0,(2**dim,1))
    G = cvx.matrix([A,AZ,AP],tc = 'd')
    h = cvx.matrix([bc,bs,bp])
    s = cvx.solvers.lp(c,G,h)
    mu = s['x']
    print(mu.T)		
    score = ComputeScore(X,Y,K,dim,mu,ChoMetric,p)
    return score

#
#K = 5#K for knn
#X,Y = glass_data()
##reduce the dimension
#PCAK = 8
#if  (len(X[0])>PCAK ) :
#    pca = PCA(n_components=PCAK)
#    X = pca.fit_transform(X)	


#s0 = clock()

#Mah_score = ComputeKNNScore(X,Y,K,1)	
#s1 = clock()
#print('OrgKNN p1 time is ',s1-s0)
#s0 = s1
#
#Eud_score = ComputeKNNScore(X,Y,K,2)	
#s1 = clock()
#print('OrgKNN p2 time is ',s1-s0)
#s0 = s1
#
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
#
