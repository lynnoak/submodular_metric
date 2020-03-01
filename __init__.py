import time
from sklearn.decomposition import PCA
from metric_learn import *
from src.mydatasets import *
from genutils import *

import pprint

"""
test
"""

mydata = [data_balance(),data_liver(),data_seeds(),data_glass(),data_iono()]#,data_sonar(),data_digits(),data_segment()]
dataname = ["balance","liver","seeds","glass","iono"]#,"sonar","digits","segment"]

#list_num_constraints = [1,2,3,4,5]

num_constraints = 100
S = {}

for i in range(len(mydata)):
#    for num_constraints in list_num_constraints:
        X, Y = mydata[i]
        K = 5
        PCAK = 10
        if (len(X[0]) > PCAK):
            pca = PCA(n_components=PCAK)
            X = pca.fit_transform(X)
        style = 0
        p = 2
        mu = ChoqQP(X, Y, style, p, num_constraints)
        Chq_2_score = ColKNNScoreSM(X, Y, K, mu, p, title="Sub. p2 ")
        S[dataname[i]+str(num_constraints)] = Chq_2_score

saveout = sys.stdout
file = open('output.txt', 'a')
sys.stdout = file
print("\n\n\n-------------------\n")
print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+"\n")
print("\n-------------------\n\n")
pprint.pprint(S, width=1)
print("\n\n\n-------------------\n\n")
file.close()
sys.stdout = saveout

"""

i = 0
X,Y = mydata[i]

S = {}
K = 5#K for knn3
num_constraints=10

PCAK = 10
if  (len(X[0])>PCAK ) :
    pca = PCA(n_components=PCAK)
    X = pca.fit_transform(X)

S['Euc.'] = ColKNNScore(X,Y,K,2,scoring = 'acc',title="Euc.")


try: 
    s0 = time.clock()
    lmnn = LMNN()
    lmnn.fit(X,Y)
    XL = lmnn.transform(X)
    s1 = time.clock()
    ttt = s1-s0
    print("time lmnn:" + str(ttt))
    S_LMNN = ColKNNScore(XL, Y, K, 2, scoring = 'acc',title="LMNN")
except:
    S_LMNN = 0
    ttt = 0
    print('LMNN error')
S['LMNN'] =  S_LMNN
S['LMNN_time'] = ttt

try:
    s0 = time.clock()
    itml = ITML_Supervised(num_constraints=num_constraints)
    itml.fit(X, Y)
    XI = itml.transform(X)
    s1 = time.clock()
    ttt = s1-s0
    print("time itml:" + str(ttt))
    S_ITML = ColKNNScore(XI, Y, K, 2, scoring = 'acc',title="ITML")
except:
    S_ITML = 0
    ttt = 0
    print('ITML error')
S['ITML'] =  S_ITML
S['ITML_time'] = ttt

try:
    s0 = time.clock()
    try:
        lfda = LFDA(k=3)
        lfda.fit(X, Y)
    except:
        lfda = LFDA(dim = int(0.9*(len(X[0]))),k=3,num_constraints=num_constraints)
        tX = X + 10 ** -4
        lfda.fit(tX, Y)
    XL = lfda.transform(X)
    s1 = time.clock()
    ttt = s1-s0
    print("time lfda:" + str(ttt))
    S_LFDA = ColKNNScore(XL, Y, K, 2, scoring = 'acc',title="LFDA")
except:
    print('LFDA error')
    S_LFDA = 0
    ttt = 0
S['LFDA'] =  S_LFDA
S['LFDA_time'] = ttt


try:
    s0 = time.clock()
    lsml = LSML_Supervised(num_constraints=num_constraints)
    try:
        lsml.fit(X, Y)
    except:
        tX = X + 10 ** -4
        lsml.fit(tX, Y)
    XL = lsml.transform(X)
    s1 = time.clock()
    ttt = s1-s0
    print("time lsml:" + str(ttt))
    S_LSML = ColKNNScore(XL, Y, K, 2, scoring = 'acc',title="LSML")
except:
    print('LSML error')
    S_LSML = 0
    ttt = 0
S['LSML'] =  S_LSML
S['LSML_time'] = ttt

#style = 0
#p = 1
#s0 = clock()
#mu = ChoqQP(X,Y,style,p)
#s1 = clock()
#Chq_1_score = ComputeScore(X, Y, K, mu, p, "Sub. p1 ")
#ttt = s1-s0
#print("time SubKNN p1:" + str(ttt))


style = 0
p = 2
s0 = time.clock()
mu = ChoqQP(X,Y,style,p,num_constraints)
s1 = time.clock()
Chq_2_score = ColKNNScoreSM(X, Y, K, mu,p,title="Sub. p2 ")
ttt = s1-s0
print("time SubKNN p1:" + str(ttt))

S['L_f'] =  Chq_2_score
S['L_f_time'] = ttt

#Chq_k_score = []
#Chq_k_t = []
#for style in range(1,len(X[0])+1):
#    s0 = clock()
#    mu = ChoqQP(X,Y,style,p=2)
#    s1 = clock()
#    Chq_k_t.append(s1-s0)
#    Chq_k_score.append(ComputeScore(X, Y, K, mu, pnorm=2,title=str(style)+" add "))
#    ttt = s1-s0
#    print(style,'-add time:',s1-s0)

"""



"""

def myRes (X,Y,num_constraints = 100,K = 5,PCAK=10):
    S = {}

    if (len(X[0]) > PCAK):
        pca = PCA(n_components=PCAK)
        X = pca.fit_transform(X)

    S_Euc = ColKNNScore(X, Y, K, 2, scoring='acc', title="Euc.")
    S['Euc.'] = format(S_Euc['acc'][0],'.2%')+"(+/-)"+format(S_Euc['acc'][1],'.2%')


#    try:
#        s0 = time.clock()
#        lmnn = LMNN()
#        lmnn.fit(X,Y)
#        XL = lmnn.transform(X)
#        s1 = time.clock()
#        ttt = s1-s0
#        print("time lmnn:" + str(ttt))
#        S_LMNN = ColKNNScore(XL, Y, K, 2, scoring = 'acc',title="LMNN")
#    except:
#        S_LMNN = 0
#        ttt = 0
#        print('LMNN error')
#    S['LMNN'] =  S_LMNN
#    S['LMNN_time'] = ttt    

    try:
        s0 = time.clock()
        itml = ITML_Supervised(num_constraints=num_constraints)
        itml.fit(X, Y)
        XI = itml.transform(X)
        s1 = time.clock()
        ttt = s1-s0
        print("time itml:" + str(ttt))
        S_ITML = ColKNNScore(XI, Y, K, 2, scoring = 'acc',title="ITML")
    except:
        S_ITML = 0
        ttt = 0
        print('ITML error')
    S['ITML'] = format(S_ITML['acc'][0],'.2%')+"(+/-)"+format(S_ITML['acc'][1],'.2%')
    S['ITML_time'] = format(ttt,'.4g')

    try:
        s0 = time.clock()
        try:
            lfda = LFDA(k=3)
            lfda.fit(X, Y)
        except:
            lfda = LFDA(dim = int(0.9*(len(X[0]))),k=3,num_constraints=num_constraints)
            tX = X + 10 ** -4
            lfda.fit(tX, Y)
        XL = lfda.transform(X)
        s1 = time.clock()
        ttt = s1-s0
        print("time lfda:" + str(ttt))
        S_LFDA = ColKNNScore(XL, Y, K, 2, scoring = 'acc',title="LFDA")
    except:
        print('LFDA error')
        S_LFDA = 0
        ttt = 0  
    S['LFDA'] = format(S_LFDA['acc'][0],'.2%')+"(+/-)"+format(S_LFDA['acc'][1],'.2%')
    S['LFDA_time'] = format(ttt,'.4g')    

    try:
        s0 = time.clock()
        lsml = LSML_Supervised(num_constraints=num_constraints)
        try:
            lsml.fit(X, Y)
        except:
            tX = X + 10 ** -4
            lsml.fit(tX, Y)
        XL = lsml.transform(X)
        s1 = time.clock()
        ttt = s1-s0
        print("time lsml:" + str(ttt))
        S_LSML = ColKNNScore(XL, Y, K, 2, scoring = 'acc',title="LSML")
    except:
        print('LSML error')
        S_LSML = 0
        ttt = 0
    S['LSML'] = format(S_LSML['acc'][0],'.2%')+"(+/-)"+format(S_LSML['acc'][1],'.2%')
    S['LSML_time'] = format(ttt,'.4g')    

    try:
        style = 0
        p = 2
        s0 = time.clock()
        mu = ChoqQP(X,Y,style,p,num_constraints)
        s1 = time.clock()
        L_f = ColKNNScoreSM(X, Y, K, mu,p,title="Sub. p2 ")
        ttt = s1-s0
        print("time Sub p1:" + str(ttt)) 
    except:
        print('Sub error')
        L_f = 0
        ttt = 0

    S['L_f'] = format(L_f['acc'][0],'.2%')+"(+/-)"+format(L_f['acc'][1],'.2%')
    S['L_f_time'] = format(ttt,'.4g')  

    return S



def myRes (X,Y,num_constraints = 100,K = 5,PCAK=10):
    S = {}

    if (len(X[0]) > PCAK):
        pca = PCA(n_components=PCAK)
        X = pca.fit_transform(X)

    for style in range(2,len(X[0])+1):
        s0 = time.clock()
        mu = ChoqQP(X, Y, style, p=2, num_constraints = num_constraints)
        s1 = time.clock()
        ttt = s1-s0
        print(style,'-add time:',ttt)
        S[style] = str(style)+'-add time:'+str(ttt)

    return S

#mydata = [data_balance(),data_seeds(),data_glass(),data_iono(),data_digits(),data_segment(),data_liver(),data_sonar()]
#dataname = ["balance","seeds","glass","iono","digits","segment","liver","sonar"]

mydata = [data_balance(),data_seeds(),data_glass(),data_iono(),data_segment()]
dataname = ["balance","seeds","glass","iono","segment"]

#list_num_constraints = [350,400,450,500]
num_constraints =200

for i in range(len(mydata)):
#    for num_constraints in list_num_constraints:
        X,Y = mydata[i]
        S = myRes(X,Y,num_constraints)
        saveout = sys.stdout
        file = open('output.txt','a')
        sys.stdout = file       
        print("\n\n\n-------------------\n")
        #print (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())+"\n")
        print("Dataset "+ dataname[i]+"\n")
        print(num_constraints)
        print("\n-------------------\n\n")
        pprint.pprint(S,width=1)
        print("\n\n\n-------------------\n\n")
        file.close()
        sys.stdout = saveout
"""


