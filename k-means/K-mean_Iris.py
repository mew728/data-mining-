from typing import Type
from sklearn.cluster import KMeans
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
kmeans = KMeans(n_clusters=3, random_state=0)
Pred = kmeans.fit_predict(X)
# print("Pred：\n",Pred)
#cm
contingency_matrix = metrics.cluster.contingency_matrix(Pred,Y) #/ Pred類數放前
print("分佈=\n",contingency_matrix)#分佈
# Purity
def purity(Pred,Y):
    return np.sum((np.sum(contingency_matrix,axis=1)/Iris.shape[0])*(np.amax(contingency_matrix[:],axis=1)/np.sum(contingency_matrix,axis=1)))
print("Purity=",purity(Pred,Y))
# Time complexity
TC = Iris.shape[0]*kmeans.n_clusters*(Iris.shape[1]-1)*kmeans.max_iter #nKdI
print("Time complexity=",TC)
#plot
plt.imshow(contingency_matrix,interpolation='none',cmap='Blues')
for (i, j), z in np.ndenumerate(contingency_matrix):
    plt.text(j, i, z, ha='center', va='center')
plt.ylabel("cluster")
plt.xlabel("truth label")
plt.xticks(Y)
plt.yticks(Pred)
plt.show()
