from sklearn.cluster import KMeans
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
# 
kmeans = KMeans(n_clusters=32, random_state=0)
Pred = kmeans.fit_predict(X)
Type, counts = np.unique(Pred, return_counts=True) 
#cm
contingency_matrix = metrics.cluster.contingency_matrix(Pred,Y) #/ Pred類數放前
# Purity
def purity(Pred,Y):
    return np.sum((np.sum(contingency_matrix,axis=1)/Wine.shape[0])*(np.amax(contingency_matrix[:],axis=1)/np.sum(contingency_matrix,axis=1)))
print("Purity=",purity(Pred,Y))
# SSE
SSE = kmeans.inertia_
print("SSE=",SSE)
# Entropy
ent=np.zeros((32,3))
def entropy(Pred,Y):
    for xx in range(0,32):
        for y in range(0,3):
            p=contingency_matrix[xx][y]/np.sum(contingency_matrix[xx])
            try:
                with np.errstate(divide='raise'):
                    ent[xx][y]=-(p*np.log2(p))
            except:
                    ent[xx][y]=0
    return np.sum(np.sum(ent,axis=1)*(np.sum(contingency_matrix,axis=1)/Wine.shape[0]))
print("Entropy=",entropy(Pred,Y))
# Time complexity
TC = Wine.shape[0]*kmeans.n_clusters*(Wine.shape[1]-1)*kmeans.max_iter #nKdI
print(TC)
