
cpdef int sign(int a):
	return(a < 0)
cpdef int signD(double a):
	return(a < 0)
cpdef double Bisection(double a, double b, func, int itmax):
	cdef double u,v,c,w
	cdef int i=itmax
	u=func(a)
	v=func(b)
	assert(signD(u)!=signD(v))
	while (i>0):
		i=i-1
		c = (a+b)/2.0
		w = func(c)
		if ((b-a)<1.0e-10):
			break
		if (signD(u)==signD(w)):
			u=w
			a=c
		else:
			v=w
			b=c
	return (a+b)/2.0
cpdef double Min(double a, double b):
	if (a<b):
		return a
	else:
		return b
cpdef double Max(double a, double b):
	if (a<b):
		return b
	else:
		return a
cpdef int IsOdd(int i):
	if (i & 0x1):
		return 1
	else:
		return 0

cpdef int Card(int A):
	cdef int c=0
	cdef int t=A
	while (t>0):
		if (t & 0x1):
			c=c+1
		t = t>>1
	return c

cpdef RemoveFromSet(A, int i):
	
	return A
#	RemoveFromSet