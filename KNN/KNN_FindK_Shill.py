from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
Bi = pd.read_csv("DM_HW2/Shill Bidding Dataset.csv", encoding='unicode_escape')
scaler.fit(Bi)
scaled_features = scaler.transform(Bi)
w= pd.DataFrame(scaled_features)
X=w.loc[:,:8]
y=w.loc[:,9]
print(w)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
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