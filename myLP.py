# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 09:36:32 2016

@author: victor
"""

from mytest import *
from cvxpy import *
import numpy




def myloss(x,A1,A2,B1,B2,dim,Fx,pnorm):
    v =[]
    for i in range(2**dim):
        v.append(x[i].value)
    v = array(v,dtype=double)
    loss = 0
    for i in range(len(A1)):
        t = 1+myChoMetric(A1[i],A2[i],dim = dim,v = v, Fx = Fx, pnorm = pnorm)
        t = t-myChoMetric(B1[i],B2[i],dim = dim,v = v, Fx = Fx, pnorm = pnorm)
        loss= loss+t        
    return loss
    
myLoadData=[seeds_data(),sonar_data(),iono_data(),iris_data(),wine_data()]

SKNN = SKNN1
stdSKNN = stdSKNN1
SLearn = []
stdSLearn = []
    
    
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
 
            
    n = len(X[0])
    m = len(X)
    rm = 100    

    index = floor(random(m)*m)
    j = 0
    A1 = []
    A2 = []
    while(len(A1)<rm):
        j = j+1
        i = j%m
        if(C[i]==C[index[i]]):
            A1.append(list(X[i,:]))
            A2.append(list(X[index[i],:]))        
    A1 = array(A1,dtype=double)
    A2 = array(A2,dtype=double) 

    index = floor(random(m)*m)
    B1 = []
    B2 = []
    while(len(B1)<rm):
        j = j+1
        i = j%m
        if(C[i]!=C[index[i]]):
            B1.append(list(X[i,:]))
            B2.append(list(X[index[i],:]))        
    B1 = array(B1,dtype=double)
    B2 = array(B2,dtype=double)
            
                    
    #parameter
    a = 0.5
    dim = n
    K = 4
    Fx = [lambda x:x]*dim
    pnorm = 1
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
            for q in d1:                
                if q&p == q:
                    v[p] = max(v[p],v[q])
    v = [i/v[pow(2,dim)-1] for i in v]
    v = array(v,dtype=double)   

    # Construct the problem.
    x = []
    for i in range(2**dim):
        x.append(Variable())
        x[i].save_value(v[i])
    objective = Minimize(myloss(x,A1,A2,B1,B2,dim,Fx,pnorm))    

    #when dim = 4
    constraints = []
    for i in range(2**dim):
        constraints+=[x[i]>=0,x[i]<=1]
        for j in range(2**dim):
            constraints+=[x[i]+x[j]-x[i&j]-x[i|j]>=0]   

    #m = Mobius(x.value)             
    #for i in range(2**n):
    #    if(bitCount(i)>=2):
    #        t = 0
    #        for j in range(i):
    #            if(j&i==i and bitCount(j)>=2):
    #                t = t+m[j]
    #        constraints.append(t<=0)
                
                 
    prob = Problem(objective, constraints)  

    print("Optimal value", prob.solve())
    print("Optimal var")
    v = [i.value for i in x]
    print(v) # A numpy matrix.  
    

    v = array(v,dtype=double)
    m,s = myComputeScore(X,C,K,dim,v,Fx,myChoMetric,pnorm)
    SLearn.append(m)
    stdSLearn.append(s)

print(SLearn)
print(stdSLearn)


title = "Lovasz metric after learning"  
nData = len(SKNN)
fig, ax = plt.subplots()
index = np.arange(nData)
bar_width = 0.3
opacity = 0.4
error_config = {'ecolor': '0.3'}
rects1 = plt.bar(index, SKNN, bar_width,
                 alpha=opacity,
                 color='g',
                 error_kw=error_config,
                 yerr = stdSKNN, 
                 label='SKNN')  

rects2 = plt.bar(index + bar_width, SLearn, bar_width,
                 alpha=opacity,
                 color='r',
                 error_kw=error_config,
                 yerr = stdSLearn, 
                 label='SLearn')  
               
plt.xlabel("Datasets")
plt.ylabel("Score of cross validation")
plt.title(title)
plt.xticks(index + bar_width, ('seeds','sonar','iono','iris','wine'))
plt.xlim(0,nData+2)
plt.legend()
plt.tight_layout()
plt.savefig(title+".png")
plt.show()


