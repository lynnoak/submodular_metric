
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