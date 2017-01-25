# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 14:28:14 2017
Submodular tool
@author: victor
"""
from genutils import *
from base import *
from sublearning import *
import numpy as np
import timeit
import cvxopt as cvx
from itertools import *
from math import *


def GenrateSortedConstraints(V):
    A1 = abs(V[0])
    A2 = abs(V[1])
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
  

def submodular(m):
    """
    Input: m = 2^dim. 
    Output: matrix with v(AuB) + v(AnB) - v(A) - v(B) for all A,B \in N
    """
    x = []
    I = []
    J = []
    j = 0
    for i in range(1,int(log2(m))):
        combs = combinations([p for p in range(1,m) if bitCount(p) == i ],2)      # change to get different k-monotonicity
        for (seta,setb) in combs:
                x.extend([1,1,-1,-1])
                I.extend([j,j,j,j])
                J.extend([seta | setb, seta & setb, seta, setb])
                j = j+1
    A = cvx.spmatrix(x,I,J)                    
    return A   

def monotonicity(m):
    """
    Input: m = 2^dim. Output: matrix with v(A) - v(B) for all B \in A \in N
    """
    x = []
    I = []
    J = []
    j = 0
    for supset in range(m):
        for subset in range(supset):
            if (supset & subset == subset) & (bitCount(supset) - bitCount(subset) == 1):
                x.extend([1,-1])
                I.extend([j,j])
                J.extend([supset,subset])
                j = j+1
    A = cvx.spmatrix(x,I,J)
    return A

def k_additivity(m,k):
    """
    Input: m=2^dim, k \in [1,dim] 
    Output: mobius expressions for all A \in N, |A| > k
    """
    x = []
    I = []
    J = []
    j = 0
    for supset in range(m):
        if (bitCount(supset) > k):
            for subset in range(supset):
                if (supset & subset == subset):
                    x.extend([pow(-1,bitCount(supset-subset))])
                    I.extend([j])
                    J.extend([subset])
            x.extend([1])
            I.extend([j])
            J.extend([supset])
            j = j+1
    A = cvx.spmatrix(x,I,J)
    return A  
    
 