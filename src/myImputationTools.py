import numpy as np
from sklearn import *
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import KNNImputer

data= datasets.load_iris()
X = data['data']
y = data['target']

def data_mcar(X,y):
    (xi,xj) = X.shape
    n = int(xi*xj/5)
    i = np.random.randint(xi,size=n)
    j = np.random.randint(xj,size=n)
    X[i,j] = np.nan
    return X,y

def data_mnar(X,y):
    yi = [i for i in range(len(y)) if y[i]==y[0]]
    (xi, xj) = X.shape
    n = int(len(yi)*xj/5)
    i = np.random.randint(len(yi),size=n)
    i = yi[i]
    j = np.random.randint(xj,size=n)
    X[i, j] = np.nan
    return X,y

X_nan,y = data_mcar(X,y)

class Imputer(object):
    """
   Imputation for the missing value
    :param X_nan: feature of dataset with missing value(np.nan)
    :param alg: imputation algorithm for the first learnt submodular algorithm.
        alg = ['Listwise',
        'SimpleImputer_mean','SimpleImputer_median','SimpleImputer_most_frequent',
        'IterativeImputer','KNNImputer']
    :param X_imp: the data after imputation
    :param error_imp: the error dist
    """
    X_imp = {}
    y_imp = {}
    error_imp ={}

    def __init__(self,X = X_nan,y =y,X_train=None,alg =['Listwise'],metric = 'nan_euclidean'):
        self.X = X
        X_original = X
        index_listwise = [i for i in range(X[:, 0].size) if sum(np.isnan(X[i, :])) == 0]
        X_listwise = X[index_listwise, :]
        y_listwise = y[index_listwise]
        if X_train ==None:
            self.X_train = X_original
        else:
            self.X_train = X_train
        self.metric =metric

        test = ['Listwise','SimpleImputer_mean','IterativeImputer','KNNImputer']
        allalg = ['Listwise','SimpleImputer_mean','SimpleImputer_median','SimpleImputer_most_frequent','IterativeImputer','KNNImputer']
        if alg =='test' or alg ==['test']:
            alg = test
        if alg == 'all' or alg == ['all']:
            alg = allalg
        if not isinstance(alg, list):
            alg = [alg]
        if alg==[] or not set(allalg) >= set(alg):
            alg = ['Listwise']
        self.alg = alg

        ditalg = {"SimpleImputer_mean": SimpleImputer(strategy='mean'),
                  "SimpleImputer_median": SimpleImputer(strategy= 'medien'),
                  "SimpleImputer_most_frequent": SimpleImputer(strategy='most_frequent'),
                  "IterativeImputer": IterativeImputer(),
                  "KNNImputer": KNNImputer(metric = self.metric)}

        for i in self.alg:
            if i =='Listwise':
                self.X_imp[i] = X_listwise
                self.y_imp[i] = y_listwise
                self.error_imp[i] = 0
            else:
                algmodel = ditalg[i]
                try:
                    algmodel.fit(self.X_train)
                    self.X_imp[i] = algmodel.transform(X)
                    self.y_imp[i] = y
                    self.error_imp[i] = 0
                except:
                    print(i+" get error!\n")
                    self.error_imp[i] = 1
                    self.X_imp[i] = X_listwise
                    self.y_imp[i] = y_listwise

    def GetPara(self):
        print(self.alg)
        print(self.X_imp)
        print(self.error_imp)
        print(self.metric)
        return self.__dict__

