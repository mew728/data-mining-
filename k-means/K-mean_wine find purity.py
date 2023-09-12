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
pur=[]
sse=[]
entr=[]
num=[]
time=[]
for numc in range (1,40):
    kmeans = KMeans(n_clusters=numc, random_state=0)
    Pred = kmeans.fit_predict(X)
    Type, counts = np.unique(Pred, return_counts=True) 
    #cm
    contingency_matrix = metrics.cluster.contingency_matrix(Pred,Y) #/ Pred類數放前
    # Purity
    def purity(Pred,Y):
        return np.sum((np.sum(contingency_matrix,axis=1)/Wine.shape[0])*(np.amax(contingency_matrix[:],axis=1)/np.sum(contingency_matrix,axis=1)))
    pur.append(purity(Pred,Y))
    # SSE
    SSE = kmeans.inertia_
    sse.append(SSE)
    # Entropy
    ent=np.zeros((numc,3))
    def entropy(Pred,Y):
        for xx in range(0,(numc-1)):
            for y in range(0,3):
                p=contingency_matrix[xx][y]/np.sum(contingency_matrix[xx])
                try:
                    with np.errstate(divide='raise'):
                        ent[xx][y]=-(p*np.log2(p))
                except:
                    ent[xx][y]=0
        return np.sum(np.sum(ent,axis=1)*(np.sum(contingency_matrix,axis=1)/Wine.shape[0]))
    entr.append(entropy(Pred,Y))
    num.append(numc)
    # Time complexity
    TC = Wine.shape[0]*kmeans.n_clusters*(Wine.shape[1]-1)*kmeans.max_iter #nKdI
    time.append(TC)
# 
plt.subplot(2,2 ,1)
plt.plot(num,pur ,'r-x')   
plt.xlabel('# of cluster')
plt.ylabel('Purity')
plt.title("Clusters and Purity")  
# 
plt.subplot(2,2 ,3)
plt.plot(num,sse ,'r-x')     
plt.xlabel('# of cluster')
plt.ylabel('SSE')
plt.title("Clusters and SSE")   
# 
plt.subplot(2,2 ,2)
plt.plot(num,entr ,'r-x')      
plt.xlabel('# of cluster')
plt.ylabel('Entrpoy')
plt.title("Clusters and Entrpoy")  
# 
plt.subplot(2,2 ,4)
plt.plot(num,time ,'r-x')      
plt.xlabel('# of cluster')
plt.ylabel('Time complexity')
plt.title("Clusters and Time complexity")  
plt.tight_layout()
plt.show()
#'''