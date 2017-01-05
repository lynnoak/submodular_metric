from genutils import *
from base import *
from sublearning import *
import numpy as np
import timeit

from capacity_parameters import *
from mydatasets import *
from math import log2
import cvxopt as cvx
from itertools import *


def mafonction(x):
	return x*x*x - x*x - 2


def timer(call):
	start_time = timeit.default_timer()
	call
	print(timeit.default_timer() - start_time)


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

X,Y = seeds_data()

dim = len(X[0])

number_of_constraints = 10
max_iter = 30
A=GenerateTriplets(Y,number_of_constraints, max_iter)
print(A)
V =GenerateConstraints(A,X)
print(V)


    
A = GenrateSortedConstraints(V)
print(A)

#    
#AZ = convexity(2**dim)
#print(AZ)
#A = np.row_stack((AZ,A))
#b = np.hstack((np.zeros(dim),np.ones(number_of_constraints)))
#print(b)

P = np.zeros(dim)
q = np.ones(dim)


cvx.solvers.qp(P,q,A,b)
