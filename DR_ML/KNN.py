from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import metrics

iris = datasets.load_iris()
iris_x = iris.data
iris_y = iris.target
print('iris_x=', iris_x)
print('iris_y=', iris_y)
X_train, X_test, y_train, y_test = train_test_split(iris_x, iris_y, test_size=0.2, random_state=42)
# 定义模型
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
# knn.fit(X_train,y_train.ravel())
# 预测
y_pred_on_train = knn.predict(X_test)
# 输出
print(y_pred_on_train)
print("------------")
print(y_test)
acc = metrics.accuracy_score(y_test, y_pred_on_train)
print(acc)
