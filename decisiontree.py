from sklearn import preprocessing
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.tree import export_graphviz
from io import StringIO
import pydotplus
from IPython.display import display,Image


f=open(r"C:\Users\USER\Desktop\titanic.csv")
titanic_train=pd.read_csv(f)
titanic_train=titanic_train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1)

# 將 Age 遺漏值以 median 填補
age_median = np.nanmedian(titanic_train["Age"])
new_Age = np.where(titanic_train["Age"].isnull(), age_median, titanic_train["Age"])
titanic_train["Age"] = new_Age

# 創造 dummy variables
label_encoder = preprocessing.LabelEncoder()
titanic_train["Sex"]= label_encoder.fit_transform(titanic_train["Sex"])
titanic_train["Embarked"]= label_encoder.fit_transform(titanic_train["Embarked"])

titanic_x=titanic_train.drop('Survived', axis = 1)

# 切分訓練與測試資料
train_x, test_x, train_y, test_y = train_test_split(titanic_x,titanic_train["Survived"], test_size = 0.3)


# 建立分類器
clf = tree.DecisionTreeClassifier(criterion='entropy',max_depth = 5, 
                                random_state=0)
titanic_clf = clf.fit(train_x,train_y)

# 預測
test_y_predicted = titanic_clf.predict(test_x)
#print(test_y_predicted)

# 標準答案
#print(test_y)

# 績效
accuracy = metrics.accuracy_score(test_y, test_y_predicted)
print(accuracy)

dot_data = StringIO()
export_graphviz(titanic_clf, out_file=dot_data, 
                feature_names=titanic_x.columns)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
display(Image(graph.create_png()))
