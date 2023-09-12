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
    #skl
    HC=AgglomerativeClustering(n_clusters=40,affinity='euclidean',linkage=x)
    Pred=HC.fit_predict(X)
    # print("Pred：\n",Pred)
    # cm
    contingency_matrix = metrics.cluster.contingency_matrix(Pred,Y) #/ Pred類數放前
    # print("分佈=\n",contingency_matrix)#分佈
    # Purity
    def purity(Pred,Y):
        return np.sum((np.sum(contingency_matrix,axis=1)/Wine.shape[0])*(np.amax(contingency_matrix[:],axis=1)/np.sum(contingency_matrix,axis=1)))
    print("Method Type (",x,") Purity=",purity(Pred,Y))
    # Ent
    def entropy(Pred,Y):
        ent=np.zeros((40,7))
        for xx in range(0,40):
            for y in range(0,7):
                p=contingency_matrix[xx][y]/np.sum(contingency_matrix[xx])
                try:
                    with np.errstate(divide='raise'):
                        ent[xx][y]=-(p*np.log2(p))
                except:
                        ent[xx][y]=0
        return np.sum(np.sum(ent,axis=1)*(np.sum(contingency_matrix,axis=1)/Wine.shape[0]))
    print("Method Type (",x,") Entropy=",entropy(Pred,Y))
    # method plot
    sys.setrecursionlimit(2000)
    plt.figure(1)
    plt.subplot(2,2,count) #整圖單圖
    dis=sch.linkage(X,metric='euclidean',method=x)
    sch.dendrogram(dis, orientation='top',distance_sort='descending',show_leaf_counts=True)
    plt.title(x)
    plt.xlabel('Number of Data')
    plt.ylabel('Distance')
    plt.tight_layout()
    # # cm  plot
    plt.figure(2)
    plt.subplot(2,2,count) #整圖單圖
    plt.imshow(contingency_matrix,interpolation='none',cmap='Blues')
    for (i, j), z in np.ndenumerate(contingency_matrix):
        plt.text(j, i, z, ha='center', va='center')
    plt.title(x)
    plt.ylabel("cluster")
    plt.xlabel("truth label")
    count =count +1
# Time complexity
TC = np.power(Wine.shape[0],3)#N^3
print("Time complexity=",TC)
# 
plt.tight_layout()
plt.show()