from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
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
Train=svm.SVR(C=1, kernel='linear')
train = Train.fit(X_train, y_train)
# 
pred = train.predict(X_test) 
Ans=pd.DataFrame(y_test) 
Pred=pd.DataFrame(pred)
MSE = metrics.mean_squared_error(Ans,Pred)
RMSE = np.sqrt(metrics.mean_squared_error(Ans,Pred))
MAPE= (metrics.mean_absolute_percentage_error(Ans,Pred))
print("SVR Adult績效指標\nMSE=",MSE,"RMSE=",RMSE,"MAPE=",MAPE)
 # '''


