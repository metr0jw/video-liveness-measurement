from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import mglearn

X = pd.read_csv("data_itzy.csv")
X_scaled = MinMaxScaler().fit_transform(X)

# Activeness = 1 - (SSIM12 + SSIM23 + SSIM13)/3 + (MSE12 + MSE23 + MSE13)/3
idx_size = len(X.index) - 1
activeness = list()
for idx in range(0, idx_size):
    calc = 1 + (X_scaled[idx][0] + X_scaled[idx][1] + X_scaled[idx][2])/3 - (X_scaled[idx][3] + X_scaled[idx][4] + X_scaled[idx][5])/3
    activeness.append(calc)

activeness_scaled = list()
for count in range(0, len(activeness)-29, 30):
    compression = 0
    for idx in range(0, 30):
        compression += activeness[count+idx]
    activeness_scaled.append(compression/30)
X = pd.DataFrame(activeness_scaled)

clf = AgglomerativeClustering(n_clusters=3, affinity="euclidean", linkage='ward')
clf.fit(X)
y_predict = clf.fit_predict(X)

plt.figure(figsize=(20, 20))
plt.plot(activeness_scaled)
plt.show()