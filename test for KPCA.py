from src.myMLtools import *
from src.mydatasets import *
from sklearn.decomposition import KernelPCA

mydata = [data_balance(),data_liver(),data_seeds(),data_glass(),data_iono(),data_sonar(),data_digits(),data_segment()]
dataname = ["balance","liver","seeds","glass","iono","sonar","digits","segment"]

S = {}
for i in range(len(mydata)):
    X,y = mydata[i]
    transformer = KernelPCA(n_components=min(7,len(X[0])), kernel='linear')
    X_transformed = transformer.fit_transform(X)
    knn = KNN_score()
    s = knn.ColKNNScore(X_transformed,y,title=dataname[i])
    S[dataname[i]] =s
