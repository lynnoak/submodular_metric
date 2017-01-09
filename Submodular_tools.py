# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 14:28:14 2017

@author: victor
"""

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

def GenrateSortedConstraints(V):
    A1 = V[0]
    A2 = V[1]
    n = len(A1)
    dim = len(A1[0])
    A = np.zeros((n,2**dim), dtype = np.float64)
    for i in range(n):
        pp = np.argsort(A1[i])
        perm = [int(pow(2,j)) for j in pp]
        for j in range(dim):
            A[i,sum(perm[j:])] += A1[i,pp[j]]
            A[i,sum(perm[j+1:])] -= A1[i,pp[j]]
    pp = np.argsort(A2[i])
    perm = [int(pow(2,j)) for j in pp]
    for j in range(dim):
        A[i,sum(perm[j:])] -= A2[i,pp[j]]
        A[i,sum(perm[j+1:])] += A2[i,pp[j]]
    A = cvx.matrix(A)
    return A
    
#def GenrateMLConstraints(V):
#    A1 = V[0]
#    A2 = V[1]
#    n = len(A1)
#    dim = len(A1[0])
#    A = np.zeros((n,2**dim), dtype = np.float64)
#    for i in range(n):
#        
    