# Author : Hoel Le Capitaine
import numpy as np
cimport numpy as np
cpdef GetSimPair(Y, max_iter):
	cdef int n = len(Y)
	cdef int i = np.random.randint(low=0,high=n-1)
	cdef int j
	cdef int iter = 0
	while iter<max_iter:
		iter = iter + 1
		j = np.random.randint(low=0,high=n-1)
		if (Y[i]==Y[j] and i != j):
			return i,j
	print("Did not find a corresponding sample (same label), try to modify max_iter parameter")
	return i,j
cpdef GetSimPairAs(Y,i, max_iter):
	cdef int n = len(Y)
	cdef int j
	cdef int iter = 0
	while iter<max_iter:
		iter = iter + 1
		j = np.random.randint(low=0,high=n-1)
		if (Y[i]==Y[j] and i != j):
			return j
	print("Did not find a corresponding sample (same label), try to modify max_iter parameter")
	return j
cpdef GetDiffPair(Y, max_iter):
	cdef int n = len(Y)
	cdef int i = np.random.randint(low=0,high=n-1)
	cdef int j 
	cdef int iter = 0
	while iter<max_iter:
		iter = iter + 1
		j = np.random.randint(low=0,high=n-1)
		if (Y[i]!=Y[j]):
			return i,j
	print("Did not find a corresponding sample (different label), try to modify max_iter parameter")
	return i,j
cpdef GetDiffPairAs(Y,i, max_iter):
	cdef int n = len(Y)
	#print("label : (diff)",label)
	cdef int j
	cdef int iter = 0
	while iter<max_iter:
		iter = iter + 1
		j = np.random.randint(low=0,high=n-1)
		if (Y[i]!=Y[j]):
			return j
	print("Did not find a corresponding sample (different label), try to modify max_iter parameter")
	return j
cpdef GetTriplet(Y, max_iter):
	cdef int n = len(Y)
	cdef int i = np.random.randint(low=0,high=n-1)
	cdef int j = GetSimPairAs(Y,i,max_iter)
	cdef int k = GetDiffPairAs(Y,i,max_iter)
	return (i,j,k)
cpdef GenerateTriplets(Y,n,max_iter):
	cdef np.ndarray[np.int64_t, ndim=2] A = np.zeros((n,3), dtype = np.int64)
	for p in range(0,n):
		A[p] = GetTriplet(Y,max_iter)
	return A

cpdef GenerateConstraints(A,X,l=1):
	V =((X[A[:,0]]-X[A[:,1]])**l,(X[A[:,0]]-X[A[:,2]])**l)
	return V

