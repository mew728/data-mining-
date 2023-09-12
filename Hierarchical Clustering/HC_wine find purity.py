from numpy.lib.function_base import average
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import StandardScaler
import sys
# 
scaler = StandardScaler()
Wine = pd.read_csv("DM_HW3/wine.csv", encoding='unicode_escape')
scaler.fit(Wine[:11])
scaled_features = scaler.transform(Wine)
NewIris= pd.DataFrame(scaled_features)
X = NewIris.loc[:,:11]
Y = Wine['quality']
# 
Method=['single', 'complete', 'average','ward' ] #‘ward’, ‘complete’, ‘average’, ‘single’ # ”single”, “complete”, “average”, “weighted”, “centroid”, “median”, “ward”
count=1
for x in Method:
    pur=[]
    ent=[]
    c=[]
    for numc in range(3,40):
        #skl
        HC=AgglomerativeClustering(n_clusters=numc,affinity='euclidean',linkage=x)
        Pred=HC.fit_predict(X)
        # print("Pred：\n",Pred)
        # cm
        contingency_matrix = metrics.cluster.contingency_matrix(Pred,Y) #/ Pred類數放前
        # print("分佈=\n",contingency_matrix)#分佈
        # Purity
        def purity(Pred,Y):
            return np.sum((np.sum(contingency_matrix,axis=1)/Wine.shape[0])*(np.amax(contingency_matrix[:],axis=1)/np.sum(contingency_matrix,axis=1)))
        # print("Method Type (",x,") Purity=",purity(Pred,Y))
        # Entropy
        def entropy(Pred,Y):
            ent=np.zeros((numc,7))
            for xx in range(0,(numc-1)):
                for y in range(0,7):
                    p=contingency_matrix[xx][y]/np.sum(contingency_matrix[xx])
                    try:
                        with np.errstate(divide='raise'):
                            ent[xx][y]=-(p*np.log2(p))
                    except:
                        ent[xx][y]=0
            return np.sum(np.sum(ent,axis=1)*(np.sum(contingency_matrix,axis=1)/Wine.shape[0]))
        # print("Entropy=",entropy(Pred,Y))
        pur.append(purity(Pred,Y))
        ent.append(entropy(Pred,Y))
        c.append(numc)
    plt.figure(1)
    plt.subplot(2,2,count) #整圖單圖
    plt.title(x)
    plt.plot(c,pur ,'r-x')   
    plt.xlabel('# of cluster')
    plt.ylabel('Purity')
    plt.tight_layout()
    plt.figure(2)
    plt.subplot(2,2,count) #整圖單圖
    plt.title(x)
    plt.plot(c,ent,'r-x')   
    plt.xlabel('# of cluster')
    plt.ylabel('Entropy')
    plt.tight_layout()
    count =count +1
# Time complexity
TC = np.power(Wine.shape[0],3)#N^3
print("Time complexity=",TC)
# 
plt.show()