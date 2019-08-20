from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import mglearn

X = pd.read_csv("data_itzy.csv")
X_scaled = MinMaxScaler().fit_transform(X)

X_train, X_test = train_test_split(
    X_scaled,
    train_size=0.7
)



clf = AgglomerativeClustering(n_clusters=3, affinity="euclidean", linkage='ward')
clf.fit(X_train)
assign = clf.fit_predict(X_test)

# 배열 x 오른쪽에 열 한개 추가
a = assign.reshape(-1, 1)
x1 = np.hstack([x, a])



# 각 클래스별로 데이터 추출

x_0 = x1[x1[:, 2]==0, :]
x_1 = x1[x1[:, 2]==1, :]
x_2 = x1[x1[:, 2]==2, :]



# 시각화

plt.scatter(x_0[:, 0], x_0[:, 1], cmap=mglearn.cm3)
plt.scatter(x_1[:, 0], x_1[:, 1], cmap=mglearn.cm3)
plt.scatter(x_2[:, 0], x_2[:, 1], cmap=mglearn.cm3)
plt.legend(['cluster 0', 'cluster 1', 'cluster 2'], loc=2)
plt.show()