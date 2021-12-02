# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 14:35:19 2018
@author: eric
"""

from IPython.display import display
from IPython.display import Image
import matplotlib.pyplot as plt
import pandas as pd
#由此呼叫鳶尾花資料集
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from io import StringIO
import pydotplus

#==============================================================================
#資料的前置處理
#==============================================================================
iris_dataset = datasets.load_iris() #讀取iris資料集

#iris_dataset說明：
#iris_dataset是一個類字典的結構，可以用[key]方式取值，iris_dataset裡面共有5個key，分別為
#'DESCR'：資料集說明文字(可略過)
#'data'：資料集樣本的特徵資料，其資料型態為ndarray
#'feature_names'：特徵的名字，資料型態為list
#'target'：資料集樣本的實際分類值，資料型態為ndarray
#'target_names'：分類值對應的iris種類名稱，資料型態為list

print("Feature names: \n{}" .format(iris_dataset['feature_names']))

#將資料(data與target)分成訓練集與測試集，切割比例為設定train_size為0.75(代表75%)、
#test_size為0.25%(代表25%)、random_state參數設定為0
X = iris_dataset.drop(['target', 'target_names'], axis = 1)
X_train, X_test, y_train, y_test = train_test_split(train_size = 0.75, test_size = 0.25, random_state = 0)

print("X_train 的 (行數, 欄數): {}".format(X_train.shape))
print("y_train 的 (行數, 欄數): {}".format(y_train.shape))
print("X_test 的 (行數, 欄數): {}".format(X_test.shape))
print("y_test 的 (行數, 欄數): {}".format(y_test.shape))

#建立iris_dataframe，使用dataframe資料型態可以以表格的方式紀錄每一筆樣本的特徵資料
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)

#繪出iris_dataframe的散佈矩陣圖
scatter_pic = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, 
                                         figsize=(15,15), marker='o', 
                                         hist_kwds={'bins':20})

#==============================================================================
#建立決策樹(最大樹)模型
#==============================================================================
#建立決策樹(最大樹)實例(資料型態：DecisionTreeClassifier)，初始化參數設定：
#分類方式(criterion)為entropy、最大深度(max_depth)為None、隨機狀態(random_state)為0
dt_max = **建立決策樹的實例**
**調用決策樹的適配方法(Hint:沒有回傳值)** #給定訓練資料讓模型進行配適

dt_max_training_acc = **調用決策樹計算準確度的方法(訓練)**#計算配適的準確率(預測訓練資料的準確度)
dt_max_testing_acc = **調用決策樹計算準確度的方法(測試)**#計算預測測試資料的準確度
print("Accuaracy on training set: {:.3f}".format(dt_max_training_acc))
print("Accuaracy on testing set: {:.3f}".format(dt_max_testing_acc))


print("最大樹：")
#以下是繪出最大樹的樹狀圖
dot_data = StringIO()
export_graphviz(dt_max, out_file=dot_data, 
                feature_names=iris_dataset.feature_names)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
display(Image(graph.create_png()))#顯示最大樹樹狀圖


#以下是計算最大樹在不同最大深度設定下，預測訓練/測試集的準確度
training_accuracy= **建立空的循序容器以儲存不同深度下的訓練準確度**
testing_accuracy= **建立空的循序容器以儲存不同深度下的測試準確度**
max_depth_setting = **建立可迭代物件，此物件包含1到10的整數** #最大深度從1到10
for i **迭代範圍設定**: #執行迴圈
    #建立決策樹，初始化參數設定最大深度(max_depth)為i、隨機狀態(random_state)為0
    clf = **建立決策樹實例**
    **調用決策樹的適配方法(Hint:沒有回傳值)**
    **在training_accuracy中新增此次迭代建立的模型的訓練準確度**
    **在training_accuracy中新增此次迭代建立的模型的測試準確度**


#以下是繪出上面計算的訓練/測試集準確度折線圖
plt.figure(2,figsize=(10,6))#建立figure
plt.plot(max_depth_setting, training_accuracy, label="training accuracy")#加入訓練集準確度折線圖
plt.plot(max_depth_setting, testing_accuracy, label="testing accuracy")#加入測試集準確度折線圖
plt.ylabel("Accuracy")#加入y軸的lable
plt.xlabel("n_max_depths")#加入x軸的lable
plt.legend()#加入圖例
plt.show()#顯示折線圖


#==============================================================================
#建立決策樹(最佳樹)模型
#==============================================================================
#建立決策樹(最佳樹)實例，參數設定為分類方式(criterion)為entropy、最大深度(max_depth)為3
#隨機狀態(random_state)為0
dt_best = **建立決策樹的實例**
**調用決策樹的適配方法(Hint:沒有回傳值)** #給定訓練資料讓模型進行配適


dt_best_training_acc = **調用決策樹計算準確度的方法(訓練)**#計算配適的準確率(預測訓練資料的準確度)
dt_best_testing_acc = **調用決策樹計算準確度的方法(測試)**#計算預測測試資料的準確度
print("Accuaracy on training set: {:.3f}".format(dt_best_training_acc))
print("Accuaracy on testing set: {:.3f}".format(dt_best_testing_acc))


print("最佳樹:")
#以下是繪出最大樹的樹狀圖
dot_data = StringIO()
export_graphviz(dt_best, out_file=dot_data, 
                feature_names=iris_dataset.feature_names)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
display(Image(graph.create_png()))


#==============================================================================
#比較最大樹與最佳樹的模型
#==============================================================================
#建立一個字典，字典裡面有'訓練'與'測試'兩個key，key對應的資料為一個list，
#'訓練'的key對應的list裡面依序放最大樹的訓練準確度與最佳樹的訓練準確度(注意順序不可相反)
#'測試'的key對應的list裡面依序放最大樹的測試準確度與最佳樹的測試準確度(注意順序不可相反)
model_comparison = {**填入鍵值與資料**}
result = pd.DataFrame(model_comparison, index=['最大樹', '最佳樹'])
print(result)
