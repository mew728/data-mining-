from numpy.core.fromnumeric import partition
from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import StandardScaler
# 
scaler = StandardScaler()
Wine = pd.read_csv("DM_HW3/wine.csv", encoding='unicode_escape')
scaler.fit(Wine[:11])
scaled_features = scaler.transform(Wine)
NewIris= pd.DataFrame(scaled_features)
X = NewIris.loc[:,:11]
Y = Wine['quality']
# print(X,Y)
# 
DB=DBSCAN(eps=3.4,min_samples=4) #(eps=3.1,min_samples=4)32群 (eps=3.48,min_samples=4)=0.900 ENT=0
Pred=DB.fit_predict(X)
# print("Pred：\n",Pred)
# 陣列種類數量 -> 找群數
Type, counts = np.unique(Pred, return_counts=True) 
print("類別：數量=",dict(zip(Type,counts)))
# cm
contingency_matrix = metrics.cluster.contingency_matrix(Pred,Y) #/ Pred類數放前
contingency_matrix=contingency_matrix[1::]#Clear -1
print("分佈=\n",contingency_matrix)#分佈
# Purity
def purity(Pred,Y):
    return np.sum((np.sum(contingency_matrix,axis=1)/Wine.shape[0])*(np.amax(contingency_matrix[:],axis=1)/np.sum(contingency_matrix,axis=1)))
print("Purity=",purity(Pred,Y))
# SSE
# ?
# Entropy
def entropy(Pred,Y):
    ent=np.zeros(((Type.size-1),7))
    for xx in range(0,(Type.size-1)):
        for y in range(0,7):
            p=contingency_matrix[xx][y]/np.sum(contingency_matrix[xx])
            try:
                with np.errstate(divide='raise'):
                    ent[xx][y]=-(p*np.log2(p))
            except:
                ent[xx][y]=0
    return np.sum(np.sum(ent,axis=1)*(np.sum(contingency_matrix,axis=1)/Wine.shape[0]))
print("Entropy=",entropy(Pred,Y))
# Time complexity
TC = np.power(Wine.shape[0],2)#N^2
print("Time complexity=",TC)
