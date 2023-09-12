import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
Bi = pd.read_csv("DM_HW2/Shill Bidding Dataset.csv", encoding='unicode_escape')
scaler.fit(Bi)
scaled_features = scaler.transform(Bi)
w= pd.DataFrame(scaled_features)
X=w.loc[:,:8]
y=w.loc[:,9]
# 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# 
Train=xgb.XGBRegressor(learning_rate=0.2)
train = Train.fit(X_train, y_train)
# 
pred = train.predict(X_test) 
Ans=pd.DataFrame(y_test) 
Pred=pd.DataFrame(pred)
MSE = metrics.mean_squared_error(Ans,Pred)
RMSE = np.sqrt(metrics.mean_squared_error(Ans,Pred))
MAPE= (metrics.mean_absolute_percentage_error(Ans,Pred))
print("XGBoost Shill績效指標\nMSE=",MSE,"RMSE=",RMSE,"MAPE=",MAPE)
 # '''
print('特徵重要程度: ',Train.feature_importances_)
imp, names= zip(*sorted(zip(Train.feature_importances_,X)))
plt.barh(range(len(names)),imp,align='center')
plt.yticks(range(len(names)),names)
plt.xlabel('Importance of Features')
plt.ylabel('Features')
plt.title('Importance of Each Feature')
plt.show()

