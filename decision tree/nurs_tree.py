from io import StringIO
import pandas as pd
import numpy as np
from sklearn import tree
import matplotlib as matplotlib
import matplotlib.pyplot as plt #成本測試
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# get csv
# Wind = pd.read_csv("C:\Users\Mew\Desktop\wind1.csv", encoding='unicode_escape')
Adult = pd.read_csv("DM_analyze/nursery2.csv", encoding='unicode_escape')

# X 4 features, y 4 result
features = list(Adult.columns[:6])
X = Adult[features]  
y = Adult['class']

for D in range(9,14) :
    print("max_depth=",D )
    for N in [50,60,70,80,90,100,110,120] :
        # train 80 test20
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # Tree Graph
        Train = tree.DecisionTreeClassifier(max_depth=D ,max_leaf_nodes=N, random_state=0) #node
        train = Train.fit(X_train, y_train)

        w = train.predict(X_test.sort_index())

        xx=pd.DataFrame(X_test.sort_index())
        yy=pd.DataFrame(y_test.sort_index()) #標準答案(?)
        zz=pd.DataFrame(w)
        print("nodes=",N,"->",accuracy_score(yy,zz))
        # print(accuracy_score(yy,zz))
        tree.plot_tree(train)

        # plt.show()
# inner2 = pd.concat([xx.dropna(),yy.dropna(),zz], axis=1)
# writer = pd.ExcelWriter( 'C:/Users/Mew/Desktop/33.xlsx'  , engine='xlsxwriter')
# df_S1 = pd.DataFrame(inner2)
# df_S1.to_excel(writer, sheet_name='SHEET1' ,index=False)
# writer.save()
# print('成功')
# ----------------------#

# #graph 
# from sklearn.tree import export_graphviz
# export_graphviz(train, out_file='DM_nurs/nurs_with_8_70.dot',
#                  feature_names=list(Adult.columns[:6]),
#                  class_names=train.classes_)
# # '''