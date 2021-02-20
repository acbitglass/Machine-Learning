from sklearn.datasets import load_iris   
import pandas as pd
# 朴素贝叶斯在特征是离散的情况下效果较好
from sklearn.naive_bayes import GaussianNB # 朴素贝叶斯有高斯朴素贝叶斯和伯努利朴素贝叶斯，y是二值化是伯努利效果较好
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score
import numpy as np

# 载入数据集
iris = load_iris()
# 数据
df = pd.DataFrame(iris.data, columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])
# 标签
labels = pd.DataFrame(iris.target, columns=['attribute'])
new_df = pd.concat([df, labels], axis=1)

# 划分数据集
# X_train 60% 训练集    X_validation  20% 验证集    X_test 20% 测试集
X_tt, X_validation, Y_tt, Y_validation = train_test_split(df, labels, test_size=0.2)
X_train, X_test, Y_train, Y_test = train_test_split(X_tt, Y_tt, test_size=0.25)


# 训练模型
bayes_clf = GaussianNB()
bayes_clf.fit(X_train, Y_train)
# 预测
Y_pred = bayes_clf.predict(X_validation)
# 衡量预测结果
print('验证集:')
print("ACC:", accuracy_score(Y_validation, Y_pred))
print("REC:", recall_score(Y_validation, Y_pred, average='micro'))
print("F-Score:", f1_score(Y_validation, Y_pred, average='micro'))

# 用测试集进行对比
Y_pred = bayes_clf.predict(X_test)
print('测试集:')
print("ACC:", accuracy_score(Y_test, Y_pred))
print("REC:", recall_score(Y_test, Y_pred, average='micro'))
print("F-Score:", f1_score(Y_test, Y_pred, average='micro'))

# 预测
print(bayes_clf.predict(np.array([0.1, 0.5, 0.3, 0.3]).reshape(1,-1)))