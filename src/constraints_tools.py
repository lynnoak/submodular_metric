# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 14:28:14 2017
Submodular tool
@author: victor
"""
from base import *
from sublearning import *

import numpy as np
import cvxopt as cvx
from itertools import *
from math import *

def LovaszSort(X):
    dim = len(X)
    Sor_X = np.zeros(2**dim, dtype = np.float64)
    pp = np.argsort(X)
    perm = [int(pow(2, i)) for i in pp]
    for i in range(dim):
        Sor_X[sum(perm[i:])] += X[pp[i]]
        Sor_X[sum(perm[i + 1:])] -= X[pp[i]]
    return Sor_X

def GenrateLovaszConstraints(V):
    A1 = abs(V[0])
    A2 = abs(V[1])
    n = len(A1)
    dim = len(A1[0])
    A = np.zeros((n,2**dim), dtype = np.float64)
    for i in range(n):
        A[i] = LovaszSort(A1[i])
        A[i] -= LovaszSort(A2[i])
    A = cvx.matrix(A)
    return A

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
    A = cvx.spmatrix(x,I,J,size = (max(I)+1,m))
    return A

def monotonicity(m):
    """
    Input: m = 2^dim.
    Output: matrix with v(A) - v(B) for all B \in A \in N
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
    A = cvx.spmatrix(x,I,J,size = (max(I)+1,m))
    return A


def k_additivity(m,k=2):
    """
    Input: m = 2^dim.
    Output: matrix with v(AuB) + v(AnB) - v(A) - v(B) for all A,B \in N
    """
    x = []
    I = []
    J = []
    j = 0
    k = max(k,2)
    for i in range(1,k):
        combs = combinations([p for p in range(1,m) if bitCount(p) == i ],2)      # change to get different k-monotonicity
        for (seta,setb) in combs:
                x.extend([1,1,-1,-1])
                I.extend([j,j,j,j])
                J.extend([seta | setb, seta & setb, seta, setb])
                j = j+1
    A = cvx.spmatrix(x,I,J,size = (max(I)+1,m))
    return A


"""
    Input: m=2^dim, k \in [1,dim] 
    Output: mobius expressions for all A \in N, |A| > k
"""
"""
def k_additivity(m,k):

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
    
"""

"""
Multi-linear Extension
"""

def MLSort(X):
    dim = len(X)
    Sor_X = np.zeros(2**dim, dtype = np.float64)
    X0 = np.array(X)
    for i in range(2**dim):
        bii = [1 if i & (1 << (dim-1-n)) else 0 for n in range(dim)]
        bii = np.array(bii)
        tt = bii*X0+(1-bii)*(1-X0)
        Sor_X[i] = np.prod(tt)

    return Sor_X

def GenrateMLConstraints(V):
    A1 = abs(V[0])
    A2 = abs(V[1])
    n = len(A1)
    dim = len(A1[0])
    A = np.zeros((n,2**dim), dtype = np.float64)
    for i in range(n):
        A[i] = MLSort(A1[i])
        A[i] -= MLSort(A2[i])
    A = cvx.matrix(A)
    return A

