from genutils import Card
import numpy as np
cimport numpy as np
from cvxopt import solvers
cpdef Mobius(mu):
    """
    Mobius transform for a capacity!
    """
    cdef int v = 0
    cdef int j = 0
    cdef int clen = len(mu)
    cdef np.ndarray[np.float64_t, ndim=1] m = np.zeros(clen, dtype = np.float64)
    for v in range(clen):
        for j in range(v+1):
            if v & j == j:
                m[v] += pow(-1,Card(v - j))*mu[j]   # sum (-1)^(A\B) v(B)
                if (m[v] < 0.000001) and (m[v] > -0.000001):
                    m[v] = 0.0
    return m

cpdef MuInit(int n):
    cdef int mlen = pow(2,n)
    cdef np.ndarray[np.float64_t,ndim=1] mu = np.zeros(mlen,dtype=np.float64)
    return mu



