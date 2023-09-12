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
DB=DBSCAN(eps=0.5,min_samples=4)
Pred=DB.fit_predict(X)
# print("Pred：\n",Pred)
# 陣列種類數量 -> 找群數
Type, counts = np.unique(Pred, return_counts=True) 
dictt=dict(zip(Type,counts))
print("類別：數量=",dictt)
# cm
contingency_matrix = metrics.cluster.contingency_matrix(Pred,Y) #/ Pred類數放前
contingency_matrix=contingency_matrix[1::]#Clear -1
print("分佈=\n",contingency_matrix)#分佈
# Purity
def purity(Pred,Y):
    return np.sum((np.sum(contingency_matrix,axis=1)/Iris.shape[0])*(np.amax(contingency_matrix[:],axis=1)/np.sum(contingency_matrix,axis=1)))
print("Purity=",purity(Pred,Y))

# Time complexity
TC = np.power(Iris.shape[0],2)#N^2
print("Time complexity=",TC)
#plot
plt.imshow(contingency_matrix,interpolation='none',cmap='Blues')
for (i, j), z in np.ndenumerate(contingency_matrix):
    plt.text(j, i, z, ha='center', va='center')
plt.ylabel("cluster")
plt.xlabel("truth label")
plt.xticks(Y)
plt.yticks(range(len(dictt)  -1))
plt.show()
