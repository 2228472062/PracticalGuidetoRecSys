from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# 加载数据集，这里以鸢尾花数据集为例，只选择其中的两类
data = load_iris()
X = data.data[data.target < 2]
y = data.target[data.target < 2]

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

print(X_train)
# 创建逻辑回归模型实例
model = LogisticRegression()

# 训练模型
model.fit(X_train, y_train)

# 使用模型进行预测
y_pred = model.predict(X_test)

# 输出预测准确率
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')