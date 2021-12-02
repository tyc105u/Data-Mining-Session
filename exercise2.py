import pandas as pd
import numpy  as np
#由此呼叫乳癌資料集
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import preprocessing
from sklearn import metrics
from io import StringIO #data r/w in memory
from sklearn.tree import export_graphviz

iris_dataset = datasets.load_iris()

X = iris_dataset.data
Y = iris_dataset.target
#X = iris_dataset.drop(X, Y, axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size = 0.75, test_size = 0.25, random_state = 0)

print("X_train 的 (行數, 欄數): {}".format(X_train.shape))
print("y_train 的 (行數, 欄數): {}".format(y_train.shape))
print("X_test 的 (行數, 欄數): {}".format(X_test.shape))
print("y_test 的 (行數, 欄數): {}".format(y_test.shape))

#建立iris_dataframe，使用dataframe資料型態可以以表格的方式紀錄每一筆樣本的特徵資料
iris_dataframe = pd.DataFrame(X_train, columns=iris_dataset.feature_names)

#繪出iris_dataframe的散佈矩陣圖
scatter_pic = pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15,15), marker='o', hist_kwds={'bins':20})


dt_max = DecisionTreeClassifier()
dt_max.fit(X_train, y_train)

"""
把這些參數調進DecisionTreeClassifier()
DecisionTreeClassifier() 模型方法中也包含非常多的参数值。例如：
criterion = gini/entropy 可以用来选择用基尼指数或者熵来做损失函数。
splitter = best/random 用来确定每个节点的分裂策略。支持 “最佳” 或者“随机”。
max_depth = int 用来控制决策树的最大深度，防止模型出现过拟合。
min_samples_leaf = int 用来设置叶节点上的最少样本数量，用于对树进行修剪。
"""

predict_result = dt_max.predict(X_test)
# 預測結果: predict_result; 真實結果: y_test
acc = metrics.accuracy_score(y_test, predict_result)
dt_max_testing_acc = acc
print("Accuaracy on testing set: {:.3f}".format(dt_max_testing_acc))
predict_result = dt_max.predict(X_train)
acc = metrics.accuracy_score(y_train, predict_result)
dt_max_training_acc = acc
print("Accuaracy on training set: {:.3f}".format(dt_max_training_acc))

print("最大樹：")
#以下是繪出最大樹的樹狀圖
dot_data = StringIO()
export_graphviz(dt_max, out_file=dot_data, 
                feature_names=iris_dataset.feature_names)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
display(Image(graph.create_png()))#顯示最大樹樹狀圖
