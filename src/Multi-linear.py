

from src.submodular import *

def Mul_linear(X, M):
    dim = len(X)
    ml = 0
    for i in range(2**dim):
        bii = [1 if i & (1 << (dim-1-n)) else 0 for n in range(dim)]
        bii = np.array(bii)
        tt = bii*X+(1-bii)*(1-X)
        ml += M[i]*np.prod(tt)

    return ml

def MLMetric(X1, X2, **kwargs):
    """
       define the metric with Choquet extension
       Input: pair of object and the configuration of
           "dim" number of dimension
            "M" set function
            "p" the power of the norm
       Output: the choquet metric

   """

    X = abs(X1 - X2)
    if len(X) != kwargs["dim"]:
        M = np.arange(0, pow(2, len(X)))
        M = [bitCount(i) for i in M]
        M = np.array(M, dtype=np.double)
        return Mul_linear(X, M)
    else:
        X = [abs(i) ** kwargs["p"] for i in X]
        X = np.array(X, dtype=np.double)
        r = Mul_linear(X, kwargs["M"])


from src.submodular import *


def Mul_linear(X, M):
    dim = len(X)
    ml = 0
    for i in range(2**dim):
        bii = [1 if i & (1 << (dim-1-n)) else 0 for n in range(dim)]
        bii = np.array(bii)
        tt = bii*X+(1-bii)*(1-X)
        ml += M[i]*np.prod(tt)

    return ml

def MLMetric(X1, X2, **kwargs):
    """
       define the metric with Choquet extension
       Input: pair of object and the configuration of
           "dim" number of dimension
            "M" set function
            "p" the power of the norm
       Output: the choquet metric

   """

    X = abs(X1 - X2)
    if len(X) != kwargs["dim"]:
        M = np.arange(0, pow(2, len(X)))
        M = [bitCount(i) for i in M]
        M = np.array(M, dtype=np.double)
        return Mul_linear(X, M)
    else:
        X = [abs(i) ** kwargs["p"] for i in X]
        X = np.array(X, dtype=np.double)
        r = Mul_linear(X, kwargs["M"])
        return r ** (1 / kwargs["p"])


"""
QP MultiLinear Learning functions
p is the power of norm
style is (k) for use kadd or (0) for submodular
"""

def MLQP(X, Y, style, p, num_constraints):
    X = preprocessing.scale(X)
    m = preprocessing.MinMaxScaler()
    X = m.fit_transform(X)
    dim = len(X[0])
    max_iter = min(len(X) / 2, 200)


    if style == 0:
        AZ = submodular(2 ** dim)
    else:
        AZ = k_additivity(2 ** dim, min(style, dim - 1))
    bs = cvx.matrix(0.0, (AZ.size[0], 1))

    AP = (-1) * cvx.spmatrix(1.0, range(2 ** dim), range(2 ** dim))
    bp = cvx.matrix(0.0, (2 ** dim, 1))

    if(num_constraints<100):
        num_constraints = int(max(num_constraints *AZ.size[0],2))

    A = GenerateTriplets(Y, num_constraints, max_iter)
    V = GenerateConstraints(A, X, p)
    A = GenrateMLConstraints(V)

    a = 0.3
    P = 2 * (1 - a) * cvx.spmatrix(1.0, range(2 ** dim), range(2 ** dim))
    q = cvx.matrix(a, (2 ** dim, 1))
    G = cvx.matrix([A, AZ, AP], tc='d')
    margin = 1.0
    bc = cvx.matrix(margin, (num_constraints, 1))
    h = cvx.matrix([bc, bs, bp])
    s = cvx.solvers.qp(P, q, G, h)
    mu = s['x']
    print(mu.T)

    return mu