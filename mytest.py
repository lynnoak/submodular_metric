# -*- coding: utf-8 -*-
"""
Created on Wed May 18 15:16:22 2016

@author: victor
"""

from choquet_base import Choquet,Mobius,bitCount,Ch_gradient,MobiusB,Choquet_perm
from mydatasets import *
import Choquet_toolpack as chq
from numpy import *
from numpy.random import *
import scipy.stats
import sklearn 
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import DistanceMetric
from sklearn.neighbors.ball_tree import BallTree
import matplotlib.pyplot as plt

#x is the vector of input ,v is the set function ,Gx is the tranform function
def myMultiLinear(x,v,Gx):
    """
    Multilinear calculation with functions as arguments
    """
    a = 1
    c = 0
    xlen = len(x)       
    fx = [Gx[i](x[i]) for i in range(xlen)]
    perm = [int(pow(2,i)) for i in range(xlen)]
    for i in range(int(pow(2,len(x)))):
        t = 0
        for j in perm:
            if j&i ==j:
                #a = a*fx[int(log2(j))] 
                a = a*fx[t] 
            else:
                #a = a*(1-fx[int(log2(j))])
                a = a*(1-fx[t])                  
            t+=1
        c += a*v[i]
    return c

##test for myMultiLinear     
#X,C = seeds_data()
#dim = len(X[0])   
#v = arange(0,pow(2,dim))
#v = [bitCount(i) for i in v]
#v = array(v,dtype=double)
#Fx = [lambda x:scipy.stats.norm(0.5, 0.2).pdf(x)]*dim 
#print (myMultiLinear(abs(X[1]-X[2]),v,Fx))

##define the metric with Choquet extension
def myChoMetric(X1,X2,**kwargs):
    X = abs(X1-X2)
    if len(X)!= kwargs["dim"]:
        v = arange(0,pow(2,len(X)))
        v = [bitCount(i) for i in v]
        v = array(v,dtype=double)
        Fx = [lambda x:x]*len(X)
        return Choquet(X,v,Fx)
    else:
        X = [i**kwargs["pnorm"] for i in X]
        X = array(X,dtype = double) 
        return Choquet(X,kwargs["v"],kwargs["Fx"])**(1/kwargs["pnorm"])

##define the metric with Multilinear extension
def myMultiMetric(X1,X2,**kwargs):
    X = abs(X1-X2)
    if len(X)!= kwargs["dim"]:
        v = arange(0,pow(2,len(X)))
        v = [bitCount(i) for i in v]
        v = array(v,dtype=double)
        Fx = [lambda x:x]*len(X)
        return myMultiLinear(X,v,Fx)
    else:
        X = [i**kwargs["pnorm"] for i in X]
        X = array(X,dtype = double) 
        return myMultiLinear(X,kwargs["v"],kwargs["Fx"])**(1/kwargs["pnorm"])

#compute the score
def myComputeScore(X,C,K,dim,v,Fx,metric,pnorm): 

    #data preprocessing
    X = preprocessing.scale(X)
    m = preprocessing.MinMaxScaler()
    X = m.fit_transform(X)  
       
    #KNN with the metric 
    myKNN = KNeighborsClassifier(n_neighbors=K, 
                                 metric=metric,metric_params={"dim":dim,"v":v,"Fx": Fx,"pnorm":pnorm})
    myKNN.fit(X,C)
    #test for predict
    print(myKNN.predict_proba(X[1]))
    #cross validation
    score_my = cross_validation.cross_val_score(myKNN,X,C,cv=5)
    print(score_my)
    print("Accuracy of myKNN: %0.2f (+/- %0.2f)" % (score_my.mean(), score_my.std()))  
    return score_my.mean(),score_my.std()

#show the result
def myShowBar(SKNN,stdSKNN,SCho,stdSCho,SMul,stdSMul,title = 'test'):
        nData = len(SKNN)
        fig, ax = plt.subplots()
        index = np.arange(nData)
        bar_width = 0.2
        opacity = 0.4
        error_config = {'ecolor': '0.3'}
        rects1 = plt.bar(index, SKNN, bar_width,
                         alpha=opacity,
                         color='g',
                         error_kw=error_config,
                         yerr = stdSKNN, 
                         label='SKNN')  

        rects2 = plt.bar(index + bar_width, SCho, bar_width,
                         alpha=opacity,
                         color='r',
                         error_kw=error_config,
                         yerr = stdSCho, 
                         label='SCho')  

        rects3 = plt.bar(index + 2*bar_width, SMul, bar_width,
                         alpha=opacity,
                         color='b',
                         error_kw=error_config,
                         yerr = stdSMul, 
                         label='SMul')                 
        plt.xlabel("Datasets")
        plt.ylabel("Score of cross validation")
        plt.title(title)
        plt.xticks(index + bar_width, ('seeds','sonar','iono','iris','wine'))
        plt.xlim(0,nData+2)
        plt.legend()
        plt.tight_layout()
        plt.savefig(title+".png")
        plt.show()
        
        

def myTest(set_function = 1, tran_function = 1, pnorm = 1):         
    #load data
    myLoadData=[seeds_data(),sonar_data(),iono_data(),iris_data(),wine_data()]
    nData = len(myLoadData)
    #parameter for KNN
    K = 4    
   

    if(set_function ==1):
        title = "Bitcount function"
    elif(set_function ==2):
        title = "Max function"
    elif(set_function ==3):
        title = "Sqrt function"
    else:
        title = "test"
    
    if(tran_function != 1):
        title = title+"with trans-function"
        
    title = title + " in " +str(pnorm)+" norm"
    
    SKNN = []
    stdSKNN = []
    SCho = []
    stdSCho = []
    SMul = []
    stdSMul =[]
    
    for i in myLoadData:
        X,C = i

        #reduce the instance 
        n = max(len(X),300)
        X,C = X[0:n],C[0:n]

        #reduce the dimension
        PCAK = 6
        pca = PCA(n_components=min(PCAK,len(X[0])))
        X = pca.fit_transform(X)
        dim = len(X[0])   
        chq.dim = dim 
        
        #data preprocessing
        X = preprocessing.scale(X)
        m = preprocessing.MinMaxScaler()
        X = m.fit_transform(X)  
        
        KNN = KNeighborsClassifier(n_neighbors=K,p= pnorm)
        KNN.fit(X,C)
        print(KNN.predict_proba(X[1]))
        score_KNN = cross_validation.cross_val_score(KNN,X,C,cv=5)
        print(score_KNN)
        print("Accuracy of KNN: %0.2f (+/- %0.2f)" % (score_KNN.mean(), score_KNN.std()))   
        SKNN.append(score_KNN.mean())
        stdSKNN.append(score_KNN.std()) 

        print (SKNN)
        print (stdSKNN) 


    #set of set function
        if (set_function == 1): 
            #----------solution 1------      
            v = arange(0,pow(2,dim))
            v = [bitCount(i) for i in v]
            v = array(v,dtype=double)
            #----------solution 1------
        elif(set_function == 2):
            #---------solution 2------   
            v = zeros(pow(2,dim))
            d1 = [pow(2,i) for i in range(dim)]
            for p in range(pow(2,dim)):
                if (bitCount(p)==1):
                    v[p]=1/var(X[:,int(log2(p))])
                elif (bitCount(p)==2):
                    t = zeros(len(X[:,0]))
                    for q in d1:                
                        if q&p == q:
                            t = [x+y for x,y in zip(t,X[:,int(log2(q))])]
                            v[p] = v[p]+v[q]
                    v[p] = (var(t)-v[p])/2
                    v[p] = 1/v[p]
                else :
                    for q in range(p):                
                        if q&p == q:
                            v[p] = max(v[p],v[q])
            t = max(v)
            v = [i/t for i in v]
            v = array(v,dtype=double)
            #---------solution 2------ 
        elif(set_function == 3):
            #---------solution 3------   
            v = zeros(pow(2,dim))
            d1 = [pow(2,i) for i in range(dim)]
            for p in range(pow(2,dim)):
                for q in d1:                
                    if q&p == q:
                        v[p] = v[p]+var(X[:,int(log2(q))])**2
                v[p] = sqrt(v[p])
            t = max(v)
            v = [i/t for i in v]
            v = array(v,dtype=double)       
            #---------solution 3------
        else:
            #----------solution 1------      
            v = arange(0,pow(2,dim))
            v = [bitCount(i) for i in v]
            v = array(v,dtype=double)
            #----------solution 1------

                      
                 
        
        #set of transform function
        if(tran_function == 1):
        #----------solution 1------      
            Fx = [lambda x:x]*dim 
        #----------solution 1------
        elif(tran_function == 2):
        #----------solution 2------
            Fx = [lambda x:sin(pi*x)]*dim
        #----------solution 2------
        elif(tran_function == 3):
        #----------solution 3------
            Fx = [lambda x:scipy.stats.norm(0.5, 0.2).pdf(x)]*dim        
        #----------solution 3------
        else:
            Fx = [lambda x:x]*dim

        m,s = myComputeScore(X,C,K,dim,v,Fx,myChoMetric,pnorm)
        SCho.append(m)
        stdSCho.append(s)
        m,s = myComputeScore(X,C,K,dim,v,Fx,myMultiMetric,pnorm)
        SMul.append(m)
        stdSMul.append(s)
            

        print (SCho)
        print (stdSCho)
        print (SMul)
        print (stdSMul)

    myShowBar(SKNN,stdSKNN,SCho,stdSCho,SMul,stdSMul,title)
    return SKNN,stdSKNN,SCho,stdSCho,SMul,stdSMul,title
    
#    
#[SKNN1,stdSKNN1,SCho1,stdSCho1,SMul1,stdSMul1,title1] = myTest(1,1,1)
#[SKNN2,stdSKNN2,SCho2,stdSCho2,SMul2,stdSMul2,title2] = myTest(1,1,2)
#[SKNN3,stdSKNN3,SCho3,stdSCho3,SMul3,stdSMul3,title3] = myTest(2,1,1)
#[SKNN4,stdSKNN4,SCho4,stdSCho4,SMul4,stdSMul4,title4] = myTest(2,1,2)
#[SKNN5,stdSKNN5,SCho5,stdSCho5,SMul5,stdSMul5,title5] = myTest(3,1,1)
#[SKNN6,stdSKNN6,SCho6,stdSCho6,SMul6,stdSMul6,title6] = myTest(3,1,2)
