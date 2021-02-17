from sklearn.datasets import load_iris   #导入数据集iris
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.neighbors import KNeighborsClassifier
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
# 探索性分析

# print(df.isnull().sum()) # 无缺失值

# cols = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
# k=1.5
# for col in cols:
#     q_low = df[col].quantile(q=0.25)
#     q_high = df[col].quantile(q=0.75)
#     q_interval = q_high-q_low
#     print(len(df[col][df[col]>q_high+k*q_interval][df[col]<q_low-k*q_interval])) # 大于   上四分位数+1.5*四分位间距；  小于 下四分位数-1.5*四分位间距   为异常值，结果看，无异常值

# 将数据归一化
cols =['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
for col in cols:
	# 归一化
    new_df[col]=MinMaxScaler().fit_transform(new_df[col].values.reshape(-1,1)).reshape(1,-1)[0]
    # 标准化
    # new_df[col]=StandardScaler().fit_transform(new_df[col].values.reshape(-1,1)).reshape(1,-1)[0]

# 划分数据集
# X_train 60% 训练集    X_validation  20% 验证集    X_test 20% 测试集
X_tt, X_validation, Y_tt, Y_validation = train_test_split(df, labels, test_size=0.2)
X_train, X_test, Y_train, Y_test = train_test_split(X_tt, Y_tt, test_size=0.25)


# 训练模型
knn_clf = KNeighborsClassifier(n_neighbors=3)
knn_clf.fit(X_train, Y_train)
# 预测
Y_pred = knn_clf.predict(X_validation)
# 衡量预测结果
print('验证集:')
print("ACC:", accuracy_score(Y_validation, Y_pred))
print("REC:", recall_score(Y_validation, Y_pred, average='micro'))
print("F-Score:", f1_score(Y_validation, Y_pred, average='micro'))

# 用测试集进行对比
Y_pred = knn_clf.predict(X_test)
print('测试集:')
print("ACC:", accuracy_score(Y_test, Y_pred))
print("REC:", recall_score(Y_test, Y_pred, average='micro'))
print("F-Score:", f1_score(Y_test, Y_pred, average='micro'))

# 预测
print(knn_clf.predict(np.array([0.1, 0.5, 0.3, 0.3]).reshape(1,-1)))