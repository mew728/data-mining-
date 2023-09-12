from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import StandardScaler
# 
scaler = StandardScaler()
Iris = pd.read_csv("DM_HW3/iris.csv", encoding='unicode_escape')
change = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
Iris['class'] = Iris['class'].map(change)
# scaler.fit(Iris[:4])
# scaled_features = scaler.transform(Iris)
# NewIris= pd.DataFrame(scaled_features)
# X = NewIris.loc[:,:3]
# Y = NewIris.loc[:,4]
features = list(Iris.columns[:4])
X = Iris[features]  
Y = Iris['class']
# 
Method=['single', 'complete', 'average','ward' ] #‘ward’, ‘complete’, ‘average’, ‘single’ # ”single”, “complete”, “average”, “weighted”, “centroid”, “median”, “ward”
count=1
for x in Method:
    #skl
    HC=AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage=x)
    Pred=HC.fit_predict(X)
    # print("Pred：\n",Pred)
    # cm
    contingency_matrix = metrics.cluster.contingency_matrix(Pred,Y) #/ Pred類數放前
    # print("分佈=\n",contingency_matrix)#分佈
    # Purity
    def purity(Pred,Y):
        return np.sum((np.sum(contingency_matrix,axis=1)/Iris.shape[0])*(np.amax(contingency_matrix[:],axis=1)/np.sum(contingency_matrix,axis=1)))
    print("Method Type (",x,") Purity=",purity(Pred,Y))
    # method plot
    plt.figure(1)
    plt.subplot(2,2,count) #整圖單圖
    dis=sch.linkage(X,metric='euclidean',method=x)
    sch.dendrogram(dis, orientation='top',distance_sort='descending',show_leaf_counts=True)
    plt.title(x)
    plt.xlabel('Number of Data')
    plt.ylabel('Distance')
    plt.tight_layout()
    # cm  plot
    plt.figure(2)
    plt.subplot(2,2,count) #整圖單圖
    plt.imshow(contingency_matrix,interpolation='none',cmap='Blues')
    for (i, j), z in np.ndenumerate(contingency_matrix):
        plt.text(j, i, z, ha='center', va='center')
    plt.title(x)
    plt.ylabel("cluster")
    plt.xlabel("truth label")
    plt.xticks(Y)
    plt.yticks(Pred)
    count =count +1
# Time complexity
TC = np.power(Iris.shape[0],3)#N^3
print("Time complexity=",TC)
# 
plt.tight_layout()
plt.show()
