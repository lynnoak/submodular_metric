cimport cython
import numpy as np
cimport numpy as np
import cvxopt as cvx
cdef extern from "math.h":
  double pow(double,double)
  double exp(double)
  
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
    
cpdef Choquet(x, mu):
    cdef int i = 0
    cdef double C = 0
    cdef int xlen = len(x)
#cap = list(capacity)
    pp = np.argsort(x)
    perm = [int(pow(2,i)) for i in pp]
    for i in range(xlen):
      C += x[pp[i]]*(mu[sum(perm[i:])]-mu[sum(perm[i+1:])])
    return C

cpdef ChoquetGradient(x, mu):
    cdef int i = 0
    perm = [int(pow(2,i)) for i in np.argsort(x)]
    p = []
    for i in range(len(perm)): 
        p.extend([mu[sum(perm[i:])]-mu[sum(perm[i+1:])]])
    grad_sort = [p[i]*x[i] for i in range(len(x))]
    grad = [grad_sort[i] for i in argsort(perm)]
    return grad

cpdef ChoquetPerm(x, mu):
    cdef int i = 0
    #fx = [Fx[i](x[i]) for i in range(len(x))]
    perm = [int(pow(2,i)) for i in np.argsort(x)]
    p = []
    for i in range(len(perm)): 
        p.extend([mu[sum(perm[i:])]-mu[sum(perm[i+1:])]])
    perm_vector = [p[i] for i in argsort(perm)]
    return perm_vector
    
    
cpdef GenrateSortedConstraints(V):
    cdef np.ndarray[np.float64_t, ndim=2] A = np.zeros((n,2**dim), dtype = np.float64)
    A1 = V[0]
    A2 = V[1]
    cdef int n = len(A1)
    cdef int dim = len(A1[0])
    cdef i = 0
    cdef j = 0
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
    return A