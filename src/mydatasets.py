import numpy as np
from math import floor
from sklearn import datasets, preprocessing
localrep ="./data/" 

def seeds_data():
	file=localrep+"seeds.data"
	X = np.genfromtxt(file,delimiter=",")
	x=X[:,][:,0:6]
	y=X[:,][:,7]
	return x,y
def sonar_data():
	file=localrep+"sonar.data"
	X = np.genfromtxt(file,delimiter=",")
	x=X[:,][:,0:59]
	y=X[:,][:,60]
	return x,y
def digits_data():
	digits = datasets.load_digits()
	return digits.data,digits.target

def mnist_data():
    mnist = datasets.fetch_mldata('MNIST original')
    X = mnist.data.astype(np.float64)
    Y = mnist.target 
    k = floor(len(X)/2000)
    p = []
    for i in range(2000):
        p.append(i*k+np.random.randint(low=0,high=k))
    X = X[p]
    Y = Y[p]
    return X,Y

def iono_data():
	file =localrep+"ionosphere.data"
	X = np.genfromtxt(file,delimiter=",")
	x=X[:,][:,0:33]
	y=X[:,][:,34]
	return x,y
def iris_data():
	iris = datasets.load_iris()
	return iris.data,iris.target
def balance_data():
	file =localrep+"balance-scale.data"
	X = np.genfromtxt(file,delimiter=",")
	x=X[:,][:,1:5]
	y=X[:,][:,0]
	return x,y
def wine_data():
	file =localrep+"wine.data"
	X = np.genfromtxt(file,delimiter=",")
	x=X[:,][:,1:13]
	y=X[:,][:,0]
	return preprocessing.scale(x),y
def bci_data():
	file=localrep+"breast-cancer-wisconsin.data"
	X = np.genfromtxt(file,delimiter=",")
	x=X[:,][:,1:9]
	y=X[:,][:,10]
	return x,y 
def glass_data():
	file=localrep+"glass.data"
	X = np.genfromtxt(file,delimiter=",")
	x=X[:,][:,1:9]
	y=X[:,][:,10]
	return x,y 


   