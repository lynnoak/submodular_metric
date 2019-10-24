import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from metric_learn import *
from src.mydatasets import *
from src.mytools import *

fig = plt.figure()

K = 5#K for knn3
X,Y = data_balance()
color = ['b','y','r','g','c','m','k']
color = [color[int(i)] for i in Y]

ax1 = plt.subplot(221,projection='3d')
PCAK = 3
if  (len(X[0])>PCAK) :
    pca = PCA(n_components=PCAK)
    X1 = pca.fit_transform(X)
    X1 = normalize(X1)

ax1.scatter(X1[:,0], X1[:,1], X1[:,2], c=color)
ax1.set_title("Euclidean ")


ax2 = plt.subplot(222,projection='3d')

lmnn = LMNN(k=5, learn_rate=1e-6)
lmnn.fit(X, Y)
XL = lmnn.transform(X)

PCAK = 3
if  (len(XL[0])>PCAK) :
    pca = PCA(n_components=PCAK)
    X2 = pca.fit_transform(XL)
    X2 = normalize(X2)

ax2.scatter(X2[:, 0], X2[:, 1], X2[:, 2], c=color)
ax2.set_title("LMNN")


ax3 = plt.subplot(223,projection='3d')

itml = ITML_Supervised(num_constraints=200)
itml.fit(X, Y)
XI = itml.transform(X)

PCAK = 3
if  (len(XL[0])>PCAK) :
    pca = PCA(n_components=PCAK)
    X3 = pca.fit_transform(XI)
    X3 = normalize(X3)

ax3.scatter(X3[:, 0], X3[:, 1], X3[:, 2], c=color)
ax3.set_title("ITML")

ax4 = plt.subplot(224,projection='3d')

style = 0
p = 2
mu = ChoqQP(X**2,Y,style,p)
sub_m =[ Choquet(X[i],mu) for i in range(len(X))]
transL = np.linalg.lstsq(X,sub_m)[0]
XS = X*transL


PCAK = 3
if  (len(XL[0])>PCAK) :
    pca = PCA(n_components=PCAK)
    X4 = pca.fit_transform(XS)
    X4 = normalize(X4)

ax4.scatter(X4[:, 0], X4[:, 1], X4[:, 2], c=color)
ax4.set_title("Submodular")


plt.savefig("ttt")
plt.show()

