
from src.submodular import *
from src.myImputationTools import *



X_nan,y = data_mcar(X,y)

class SubmodularWithNan (Submodular):
    """
    Learning the submodular metric which allow the miss value and return the score of the KNN
    :param X: feature of dataset with missing value(np.nan)
    """

    def __init__(self,X_nan = X_nan, y=y, imp_alg = ['Listwise'],style = 0, num_constraints =100):
        self.X_nan = X_nan
        self.y = y
        if len(imp_alg)!= 1:
            imp_alg = ['Listwise']
        self.imp_alg = imp_alg[0]
        imp = Imputer(X=self.X_nan,y=self.y,alg=imp_alg)
        self.X_imp =imp.X_imp[self.imp_alg]
        self.y_imp = imp.y_imp[self.imp_alg]

        self.style =style
        self.num_constraints =num_constraints
        super(SubmodularWithNan,self).__init__(X=self.X_imp,y=self.y_imp,style =self.style,num_constraints=self.num_constraints)
        X_var = np.var(self.X_imp,axis=0)
        self.order = np.argsort(-X_var)
        self.reorder = np.argsort(self.order)

    def GetPara(self):
        super().GetPara(self)
        print("Order:")
        print(self.order)

    def ChoMetricWithNan(self, X1, X2):
        """
        define the metric with Choquet extension
        Input: pair of object and the configuration of
        Output: the choquet metric

        """
        X = abs(X1 - X2)
        if np.isnan(X).sum() !=0:
            Xo = np.array([X[self.order[i]] for i in range(len(self.order))])
            Xo[0] = 1 if np.isnan(Xo[0]) else Xo[0]
            X[0] =Xo[0]
            X[1:] = np.array([Xo[i - 1] if np.isnan(Xo[i]) else Xo[i] for i in range(1, len(Xo))])
            X = np.array([X[self.reorder[i]] for i in range(len(self.reorder))])
        X = [abs(i) ** self.P_power for i in X]
        X = np.array(X, dtype=np.double)
        r = Choquet(X, self.SMmetric)
        r = abs(r) ** (1 / self.P_power)
        if type(r) == float:
            return r
        else:
            print(X)
            print(r)
            return 0

    def ColKNNScore(self,n_neighbors = 5, scoring = ['acc']):
        S ={}
        S[self.imp_alg] = super().ColKNNScore(n_neighbors=n_neighbors,scoring=scoring)
        if self.style ==0:
            title ="Submodular with nan"
        else:
            title = str(self.style)+" additivity with nan"

        metric = 'pyfunc'
        metric_params = {"func":self.ChoMetricWithNan}
        CKS = KNN_score(n_neighbors =n_neighbors,P_power =self.P_power, scoring=scoring,metric=metric,metric_params=metric_params)
        S['Submodular with nan'] = CKS.ColKNNScore(self.X_nan,self.y,title=title)

        return  S





