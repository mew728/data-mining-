from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
Adult_train = pd.read_csv("DM_HW2/adult.train.csv", encoding='unicode_escape')
scaler.fit(Adult_train)
scaled_features = scaler.transform(Adult_train)
w= pd.DataFrame(scaled_features)
X_train=w.loc[:,:4]
y_train=w.loc[:,5]
# 
Adult_test = pd.read_csv("DM_HW2/adult.test.csv", encoding='unicode_escape')
scaler.fit(Adult_test)
scaled_features = scaler.transform(Adult_test)
ww= pd.DataFrame(scaled_features)
X_test=ww.loc[:,:4]
y_test=ww.loc[:,5]
# 
k_number=[]
mse=[]
rmse=[]
mape=[]
for x in range(3,30):
    Train=KNeighborsRegressor(n_neighbors=x,weights='distance',algorithm='brute',p=2)
    train = Train.fit(X_train, y_train)
    # 
    pred = train.predict(X_test) 
    Ans=pd.DataFrame(y_test) 
    Pred=pd.DataFrame(pred)
    print("When k =",x)
    MSE = metrics.mean_squared_error(Ans,Pred)
    RMSE = np.sqrt(metrics.mean_squared_error(Ans,Pred))
    MAPE= (metrics.mean_absolute_percentage_error(Ans,Pred))
    print("MSE=",MSE,"RMSE=",RMSE,"MAPE=",MAPE)
    k_number.append(x)
    mse.append(MSE)
    rmse.append(RMSE)
    mape.append(MAPE)
    # '''
# mse
plt.subplot(2, 2, 1)
plt.scatter(k_number, mse, color="r",s=10, label='Predicted')
plt.xlabel('k_number')
plt.ylabel('mse')
plt.title("k-number and mse")
# rmse
plt.subplot(2, 2, 2)
plt.scatter(k_number, rmse, color="r",s=10, label='Predicted')
plt.xlabel('k_number')
plt.ylabel('rmse')
plt.title("k-number and rmse")
# 
plt.subplot(2, 2, 3)
plt.scatter(k_number, mape, color="r",s=10, label='Predicted')
plt.xlabel('k_number')
plt.ylabel('mape')
plt.title("k-number and mape")
plt.tight_layout()
plt.show()


