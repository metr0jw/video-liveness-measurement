from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv("data_bts.csv")
data_scaled = MinMaxScaler().fit_transform(data)

# Activeness = 1 - (SSIM12*Weight12 + SSIM23*W23 + SSIM13*W13)/2 + (MSE12*W12 + MSE23*W23 + MSE13*W13)/2
idx_size = len(data.index) - 1
activeness = list()
W12=0.6
W23=0.4
W13=1.0
for idx in range(0, idx_size):
    calc = 1 + (data_scaled[idx][0]*W12 + data_scaled[idx][1]*W23 + data_scaled[idx][2]*W13)/2 - (data_scaled[idx][3]*W12 + data_scaled[idx][4]*W23 + data_scaled[idx][5]*W13)/2
    activeness.append(calc)

tick = 24
activeness_scaled = list()
for count in range(0, len(activeness)-tick-1, tick):
    compression = 0
    for idx in range(0, tick):
        compression += activeness[count+idx]
    activeness_scaled.append(compression/tick)
scaler = MinMaxScaler()
activeness_scaled = scaler.fit_transform(pd.DataFrame(activeness_scaled))
X = pd.DataFrame(activeness_scaled)

clf = AgglomerativeClustering(n_clusters=3, affinity="euclidean", linkage='ward')
clf.fit(X)
y_predict = clf.fit_predict(X)
y_predict_scaled = list()
for idx in range(len(y_predict)):
    if y_predict[idx] == 0:
        y_predict_scaled.append(0.5)
    elif y_predict[idx] == 1:
        y_predict_scaled.append(1)
    elif y_predict[idx] == 2:
        y_predict_scaled.append(0)

plt.figure(figsize=(20, 20))
plt.xticks(np.arange(0, len(activeness_scaled), 5))
plt.plot(activeness_scaled, label="Activeness")
plt.legend()
plt.savefig('data_bts.png', bbox_inches='tight')
plt.plot(y_predict_scaled, ls="-.", label="Cluster")
plt.legend()
plt.savefig('prediction_bts.png', bbox_inches='tight')
pd.DataFrame(y_predict_scaled).to_csv("y_prediction_bts.csv")