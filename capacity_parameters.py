'''
Created on 11 Mar 2011

@author: gb
'''

from numpy import *
from itertools import *
from math import factorial
import random as rnd
import cvxopt as cvx
import Choquet_toolpack as chq

def convexity(m):
    """
    Input: m = 2^dim. 
    Output: matrix with v(AuB) + v(AnB) - v(A) - v(B) for all A,B \in N
    """
    x = []
    I = []
    J = []
    j = 0
    for i in range(1,int(log2(m))):
        combs = combinations([p for p in range(1,m) if chq.bitCount(p) == i ],2)      # change to get different k-monotonicity
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
            if (supset & subset == subset) & (chq.bitCount(supset) - chq.bitCount(subset) == 1):
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
        if (chq.bitCount(supset) > k):
            for subset in range(supset):
                if (supset & subset == subset):
                    x.extend([pow(-1,chq.bitCount(supset-subset))])
                    I.extend([j])
                    J.extend([subset])
            x.extend([1])
            I.extend([j])
            J.extend([supset])
            j = j+1
    A = cvx.spmatrix(x,I,J)
    return A

def limits(m):
    """
   Input: m=2^dim, Output: matrix of ones on a diagonal 
    """
    x = ones(m-2)
    I = list(range(m-2))
    J = list(range(1,m-1))   
    A = cvx.spmatrix(x,I,J,(m-2,m))
    return A

def equalities(m):
    """
    Input:m=2^dim Output: v(0)=0, v(1)=1 of a correct size
    """
    A = cvx.spmatrix([1,1],[0,1],[0,m-1],(2,m))
    return A

def shapley(m,el):
    """
    Input: m=2^dim, el \in [0,dim]. Output: Shapley coefficient vector S for a particular criterion el. <S,v> = sh(el)
    """
    el = 1 << el-1                          # convert criterion index to binary bit position, eg 5 -> 10000  (i.e. do 2^(i-1))
    x = []
    I = []
    J = []
    for i in range(m):        
        if i & el != el:
            coeff = float(factorial(chq.bitCount(m-1) - chq.bitCount(i) - 1))*factorial(chq.bitCount(i))/factorial(chq.bitCount(m-1))
            x.extend([coeff,-coeff])
            I.extend([0,0])   # for some reason it wouldn't take I=zeros(m)
            J.extend([i | el, i])
    A = cvx.spmatrix(x,I,J)
    return A  

def int_index(m,el_pair):
    """
    Input: m=2^dim, el_pair \in [0,dim]^2. Output: II coefficient vector II for a particular criterion el. <II,v> = II(el1,el2)
    """
    el_pair = tuple(1 << i-1 for i in el_pair)   # convert criterion index to binary bit position, eg 5 -> 10000  (i.e. do 2^(i-1))
    x = []
    I = []
    J = []
    for i in range(m):        
        if (i & el_pair[0] != el_pair[0]) and (i & el_pair[1] != el_pair[1]):
            coeff = float(factorial(chq.bitCount(m-1) - chq.bitCount(i) - 2))*factorial(chq.bitCount(i))/factorial(chq.bitCount(m-1)-1)
            x.extend([coeff,-coeff,-coeff,coeff])
            I.extend([0,0,0,0])   # for some reason it wouldn't take I=zeros(m)
            J.extend([i | el_pair[0] | el_pair[1], i | el_pair[0], i | el_pair[1], i])
    A = cvx.spmatrix(x,I,J)
    return A

def necessity(m,els):
    """
    Input: m=2^dim, els - list of neccessary subsets. Output: matrix with ones for A not including any of els. 1 set per row
    """
    els = [1 << i-1 for i in els]         # convert criterion index to binary bit position, eg 5 -> 10000  (i.e. do 2^(i-1))
    x = []
    I = 0
    J = []
    for i in range(1,m):        
        if ([i & p for p in els] != els):
            x.extend([1])
            I = I+1  
            J.extend([i]) 
    A = cvx.spmatrix(x,list(range(I)),J,(I,m))
    return A

def sufficiency(m,els):
    """
    Input: m=2^dim, els - list of sufficient subsets. Output: matrix with ones for A including any of els. 1 set per row
    """
    els = [1 << i-1 for i in els]         # convert criterion index to binary bit position, eg 5 -> 10000  (i.e. do 2^(i-1))
    x = []
    I = 0
    J = []
    for i in range(1,m):        
        for p in els:
            if (i & p == p):
                x.extend([1])
                I = I+1  
                J.extend([i]) 
    A = cvx.spmatrix(x,list(range(I)),J,(I,m))
    return A

def gen_inequalities(dm_info, convex=0):
    """
    Input: dim, Shapley values structure, II values structure, convexity flag
    calls the appropriate procedures
    Output: Matrix and a column, Av<=b
    """
    dim = len(dm_info['criteria_functions'])
    if convex:
        A = cvx.sparse([-convexity(2**dim),-limits(2**dim)])
        b = cvx.matrix(0,(A.size[0],1),"d")
    else:
        A = cvx.sparse([-monotonicity(2**dim),-limits(2**dim)])
        b = cvx.matrix(0,(A.size[0],1),"d")
    if 'Sh_order' in dm_info:   
        Sh_diffs = - cvx.matrix([shapley(2**dim,p[0]) - shapley(2**dim,p[1]) for p in dm_info['Sh_order']])
        b_sh_diffs = - cvx.matrix(dm_info['Sh_delta'],(Sh_diffs.size[0],1),"d")
        A = cvx.sparse([A, Sh_diffs])
        b = cvx.matrix([b,b_sh_diffs])
    if 'Sh_equal' in dm_info:
        Sh_eq1 = cvx.matrix([shapley(2**dim,p[0]) - shapley(2**dim,p[1]) for p in dm_info['Sh_equal']])
        b_sh_eq1 = cvx.matrix(dm_info['Sh_delta'],(Sh_eq1.size[0],1),"d")
        # -Sh_eq2 < delta
        Sh_eq2 = - cvx.matrix([shapley(2**dim,p[0]) - shapley(2**dim,p[1]) for p in dm_info['Sh_equal']])
        b_sh_eq2 = cvx.matrix(dm_info['Sh_delta'],(Sh_eq2.size[0],1),"d")
        A = cvx.sparse([A, Sh_eq1, Sh_eq2])
        b = cvx.matrix([b, b_sh_eq1, b_sh_eq2])
    if 'ii_values' in dm_info:
        II_vals = - cvx.matrix([int_index(2**dim,p[0]) for p in dm_info['ii_values']])
        b_ii_vals = - cvx.matrix([p[1] for p in dm_info['ii_values']])
        A = cvx.sparse([A, II_vals])
        b = cvx.matrix([b,b_ii_vals])
    if 'ii_order' in dm_info:
        print(dm_info['ii_order'])
        II_diffs = - cvx.matrix([int_index(2**dim,p[0]) - int_index(2**dim,p[1]) for p in dm_info['ii_order']])
        b_ii_diffs = - cvx.matrix(dm_info['ii_delta'],(II_diffs.size[0],1),"d")
        A = cvx.sparse([A, II_diffs])
        b = cvx.matrix([b,b_ii_diffs])
    if 'ii_positive' in dm_info:
        II_positive = -cvx.matrix([int_index(2**dim,p) for p in dm_info['ii_positive']])
        b_ii_positive = - cvx.matrix(dm_info['ii_delta'],(II_positive.size[0],1),"d")
        A = cvx.sparse([A, II_positive])
        b = cvx.matrix([b, b_ii_positive])
    if 'ii_negative' in dm_info:
        II_negative = cvx.matrix([int_index(2**dim,p) for p in dm_info['ii_negative']])
        b_ii_negative = - cvx.matrix(dm_info['ii_delta'],(II_negative.size[0],1),"d")
        A = cvx.sparse([A, II_negative])
        b = cvx.matrix([b, b_ii_negative])
    return A,b

def gen_equalities(dm_info, k_additive=0 ):
    """
    Input: dm_info(criteria functions, Shapley values structure, II values structure, Necessity criteria, Sufficiency criteria), k-additivity
    Generate matrix rows for different information classes
    Output: Matrix and a column, Aeq*v=beq
    """
    dim = len(dm_info['criteria_functions'])
    Alist = [equalities(2**dim)]
    blist = [0.,1]
    if 'Sh_values' in dm_info:
        Alist.extend([shapley(2**dim,p[0]) for p in dm_info['Sh_values']])
        blist.extend([p[1] for p in dm_info['Sh_values']])
    Aeq = cvx.sparse(Alist)
    beq = cvx.matrix(blist)
    if 'necessity' in dm_info:
        Aeq = cvx.sparse([Aeq, necessity(2**dim,dm_info['necessity'])])
        beq = cvx.matrix([beq, cvx.matrix(0.,(size(Aeq)[0]-size(beq)[0],1))])
    if 'sufficiency' in dm_info:
        Aeq = cvx.sparse([Aeq, sufficiency(2**dim,dm_info['sufficiency'])])
        beq = cvx.matrix([beq, cvx.matrix(1.,(size(Aeq)[0]-size(beq)[0],1))])
    if k_additive:
        Aeq = cvx.sparse([Aeq, k_additivity(2**dim,k_additive)])
        beq = cvx.matrix([beq, cvx.matrix(0.,(size(Aeq)[0]-size(beq)[0],1))])
    return Aeq,beq


