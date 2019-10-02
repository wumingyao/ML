from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score as f1
from sklearn.metrics import confusion_matrix as con
from sklearn import datasets

# 载入威斯康辛州乳腺癌数据
X, y = datasets.load_breast_cancer(return_X_y=True)

# 分割训练集与测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

###初始化随机森林分类器，这里为类别做平衡处理
clf = RandomForestClassifier(class_weight='balanced', random_state=1)

###返回由训练集训练成的模型对验证集预测的结果
result = clf.fit(X_train, y_train).predict(X_test)

###打印混淆矩阵
print('\n' + '混淆矩阵：')
print(con(y_test, result))

###打印F1得分
print('\n' + 'F1 Score：')
print(f1(y_test, result))

###打印测试准确率
print('\n' + 'Accuracy:')
print(clf.score(X_test, y_test))
