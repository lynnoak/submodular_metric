'''
Created on 19 Jan 2011

@author: gb
'''
cimport cython
import numpy as np
cimport numpy as np
import cvxopt as cvx
cdef extern from "math.h":
  double pow(double,double)
  double exp(double)

Fx = [lambda x: 1-exp(-3*x), lambda x: 1-exp(-3*x), lambda x: 1-exp(-3*x), lambda x: 1-exp(-3*x),lambda x: 1-exp(-3*x),lambda x: 1-exp(-x)]
dFx = [lambda x: 3*exp(-3*x), lambda x: 3*exp(-3*x), lambda x: 3*exp(-3*x), lambda x: 3*exp(-3*x),lambda x: 3*exp(-3*x),lambda x: exp(-x)]
#Fx = [lambda x: -0.029335929406647*pow(x,6) + 0.207346406139707*pow(x,5) - 0.579333438490725*pow(x,4) + 0.821527980327791*pow(x,3) - 0.698397454788673*pow(x,2)  + 0.687228652793793*x, lambda x: -0.027728205126421*pow(x,6) + 0.197460701672059*pow(x,5) - 0.567139344258168*pow(x,4) + 0.888084577320966*pow(x,3) - 1.002827533837975*pow(x,2)  + 1.152014732068982*x,lambda x: 1-exp(-3*x), lambda x: 1-exp(-3*x),lambda x: 1-exp(-3*x),lambda x: 1-exp(-3*x)]
#dFx = [lambda x: -0.029335929406647*6*pow(x,5) + 0.207346406139707*5*pow(x,4) - 0.579333438490725*4*pow(x,3) + 0.821527980327791*3*pow(x,2) - 0.698397454788673*2*x  + 0.687228652793793, lambda x: -0.027728205126421*6*pow(x,5) + 0.197460701672059*5*pow(x,4) - 0.567139344258168*4*pow(x,3) + 0.888084577320966*3*pow(x,2) - 1.002827533837975*2*x  + 1.152014732068982, lambda x: 3*exp(-3*x), lambda x: 3*exp(-3*x),lambda x: 3*exp(-3*x),lambda x: 3*exp(-3*x)]

@cython.boundscheck(False)
@cython.wraparound(False)

#cdef inline double d_min(double a, double b): return a if a <= b else b

def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)

cpdef int bitCount(int int_type) except *:
    """
    Counts bits in a binary representation of a number, or equivalently a subset power
    """
    cdef int count = 0
    while(int_type):
        int_type &= int_type - 1
        count += 1    
    return(count)     

def Mobius(np.ndarray[np.float64_t, ndim=1] capacity):
    """
    Mobuis transform for a capacity!
    """
    cdef int v = 0
    cdef int j = 0
    cdef int clen = len(capacity)
    cdef np.ndarray[np.float64_t, ndim=1] m = np.zeros(clen, dtype = np.float64)
    for v in range(clen):
        for j in range(v+1):
            if v & j == j:
                m[v] += pow(-1,bitCount(v - j))*capacity[j]   # sum (-1)^(A\B) v(B)
                if (m[v] < 0.000001) and (m[v] > -0.000001):
                    m[v] = 0.0
    return m

def MobiusB(x):
    """
    "Anti-Mobius" for a non-linear case
    Used for representation of Choquet integral as a product <v, MobB(z)> 
    """
    cdef int i = 0
    cdef int j = 0
    cdef int v = 0
    cdef int p = 0
    cdef int xlen = len(x)
    cdef int flen = int(pow(2,xlen))
    f = [0 for i in range(flen)]
    for v in range(1,flen):     # transform inverse to mobius - C = v(A) sum_{B > A} (-1)^|B\A| min_{i < B} f_i  (i.e all SUPersets)
        for j in range(v,flen+1):
            if v & j == v:
                f_set = [p for p in range(xlen) if j & (1 << p)]   # moves a bit along and checks if it is present in the integer
                f[v] += pow(-1,bitCount(j - v))*min([Fx[p](x[p]) for p in f_set])
                #if (f[v] < 0.00000001) and (f[v] > -0.00000001):
                    #f[v] = 0.0
    return cvx.matrix(f)  #CVX


def Choquet_M(np.ndarray[np.float64_t, ndim=1] x,np.ndarray[np.float64_t, ndim=1] capacity):
    """
    Calculates Choquet wrt a Mobius transform vector
    Slower than normal, especially if need to calculate Mobius 
    """
    cdef double C = 0
    cdef int i = 0
    cdef int p = 0
    m = Mobius(capacity)
    for i in range(1,long(pow(2,len(x)))):
        f_set = [p for p in range(len(x)) if i & (1 << p)]   # moves a bit along and checks if it is present in the integer
        C += m[i]*min([Fx[p](x[p]) for p in f_set])
    return C

# def Choquet(np.ndarray[np.float64_t, ndim=1] x,np.ndarray[np.float64_t, ndim=1] capacity):
#     """
#     Choquet integral calculation.
#     This must be as fast as only possible, since used many times in optimization routines
#     """
#     cdef int i = 0
#     cdef double C = 0
#     cdef int xlen = len(x)
# #cap = list(capacity)
#     fx = [Fx[i](x[i]) for i in range(xlen)]
#     pp = argsort(fx)
#     perm = [int(pow(2,i)) for i in pp]
#     for i in range(xlen):
#       C += fx[pp[i]]*(capacity[sum(perm[i:])]-capacity[sum(perm[i+1:])])
#     return C

def Choquet(np.ndarray[np.float64_t, ndim=1] x,np.ndarray[np.float64_t, ndim=1] capacity, Gx):
    """
    Choquet integral calculation with functions as arguments
    """
    cdef int i = 0
    cdef double C = 0
    cdef int xlen = len(x)
#cap = list(capacity)
    fx = [Gx[i](x[i]) for i in range(xlen)]
    pp = argsort(fx)
    perm = [int(pow(2,i)) for i in pp]
    for i in range(xlen):
      C += fx[pp[i]]*(capacity[sum(perm[i:])]-capacity[sum(perm[i+1:])])
    return C
    
#def myMultiLinear(np.ndarray[np.float64_t, ndim=1] x,np.ndarray[np.float64_t, ndim=1] capacity, Gx)
#    """
#    Multilinear calculation with functions as arguments
#    """
#    xlen = len(x)
#    fx = [Gx[i](x[i]) for i in range(xlen)]
#    perm = [int(pow(2,i)) for i in range(xlen)]
#    for i in range(int(pow(2,len(x)))):
#        for j in perm:
#            if j&i ==i:
#                a = a*fx[j]                
#            else:
#                a = a*(1-fx[j])               
#      C += a*capacity[i]
#    return C

# def Ch_gradient(np.ndarray[np.float64_t, ndim=1] x,np.ndarray[np.float64_t, ndim=1] capacity):
#     """
#     Choquet gradient. Calculates permutation and multiplies it by predefined gradients of f(z)
#     Must be very fast, since used many times in optimization routines 
#     """
#     cdef int i = 0
#     fx = [Fx[i](x[i]) for i in range(len(x))]
#     perm = [int(pow(2,i)) for i in np.argsort(fx)]
#     p = []
#     for i in range(len(perm)): 
#         p.extend([capacity[sum(perm[i:])]-capacity[sum(perm[i+1:])]])
#     dfx = [dFx[i](x[i]) for i in argsort(fx)]
#     grad_sort = [p[i]*dfx[i] for i in range(len(dfx))]
#     grad = [grad_sort[i] for i in argsort(perm)]
#     return grad

def Ch_gradient(np.ndarray[np.float64_t, ndim=1] x,np.ndarray[np.float64_t, ndim=1] capacity,Gx,dGx):
    """
    Choquet gradient. Calculates permutation and multiplies it by predefined gradients dGx
    Must be very fast, since used many times in optimization routines 
    """
    cdef int i = 0
    fx = [Gx[i](x[i]) for i in range(len(x))]
    perm = [int(pow(2,i)) for i in np.argsort(fx)]
    p = []
    for i in range(len(perm)): 
        p.extend([capacity[sum(perm[i:])]-capacity[sum(perm[i+1:])]])
    dfx = [dGx[i](x[i]) for i in argsort(fx)]
    grad_sort = [p[i]*dfx[i] for i in range(len(dfx))]
    grad = [grad_sort[i] for i in argsort(perm)]
    return grad

def Choquet_perm(np.ndarray[np.float64_t, ndim=1] x,np.ndarray[np.float64_t, ndim=1] capacity):
    """ 
    calculates a permutation vector for a given capacity and point
    """
    cdef int i = 0
    #fx = [Fx[i](x[i]) for i in range(len(x))]
    perm = [int(pow(2,i)) for i in np.argsort(x)]
    p = []
    for i in range(len(perm)): 
        p.extend([capacity[sum(perm[i:])]-capacity[sum(perm[i+1:])]])
    perm_vector = [p[i] for i in argsort(perm)]
    return perm_vector
