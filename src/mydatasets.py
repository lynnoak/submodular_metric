import numpy as np
from sklearn import datasets, preprocessing
localrep ="../data/" 

def seeds_data():
	file=localrep+"seeds.data"
	X = np.genfromtxt(file,delimiter=",")
	x=X[:,][:,0:7]
	y=X[:,][:,7]
	return x,y
def sonar_data():
	file=localrep+"sonar.data"
	X = np.genfromtxt(file,delimiter=",")
	x=X[:,][:,0:60]
	y=X[:,][:,60]
	return x,y
def digits_data():
	digits = datasets.load_digits()
	return digits.data,digits.target
def iono_data():
	file =localrep+"ionosphere.data"
	X = np.genfromtxt(file,delimiter=",")
	x=X[:,][:,0:34]
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