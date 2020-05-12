import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv("sensor_data_700.txt", delimiter=" ", header=None, names=("data","time","ir","lidar"))

# 12時から16時までのデータを抽出
d = data[ (data["time"] < 160000) & (data["time"] >= 120000) ]
d = d.loc[:, ["ir", "lidar"]]

sns.jointplot(d["ir"], d["lidar"], d, kind="kde")
plt.show()

# In[]:
print("光センサの計測値の分散:", d.ir.var())
print("LiDARの計測値の分散", d.lidar.var())

diff_ir = d.ir - d.ir.mean()
diff_lidar = d.lidar - d.lidar.mean()

a = diff_ir * diff_lidar
print("共分散:", sum(a)/(len(d-1)))

d.mean()

# In[]:
d.cov()

# In[]:
from scipy.stats import multivariate_normal

# 多次元ガウス分布のオブジェクト生成
irlidar = multivariate_normal(mean=d.mean().values.T, cov=d.cov().values)

# In[]:
import numpy as np

# 2次元平面に均等にx座標，y座標を作成
x, y = np.mgrid[0:40, 710:750]
# 40×40×2のリストを作成（xは40×40のリスト，さらにもう一次元を追加）
pos = np.empty(x.shape + (2,))
# 加えた3次元目にx,yを代入する
pos[:, :, 0] = x
pos[:, :, 1] = y
# x, y座標とそれに対応する密度を算出
cont = plt.contour(x, y, irlidar.pdf(pos))
# 等高線に値を書き込む
cont.clabel(fmt='%1.1e')

plt.show()

# In[]:
print("x座標:", x)
print("y座標:", y)

# In[]:
c = d.cov().values + np.array([[0, 20], [20,0]])
tmp = multivariate_normal(mean=d.mean().values.T, cov=c)
count = plt.contour(x, y, tmp.pdf(pos))
plt.show()
