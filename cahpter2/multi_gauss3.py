%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal
import math

x, y = np.mgrid[0:200, 0:100]
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x
pos[:, :, 1] = y

a = multivariate_normal(mean=[50,50], cov=[[50,0], [0,100]])
b = multivariate_normal(mean=[100,50], cov=[[125,0], [0,25]])
c = multivariate_normal(mean=[150,50], cov=[[100,-25*math.sqrt(3)], [-25*math.sqrt(3),50]])

for e in [a,b,c]:
    plt.contour(x, y, e.pdf(pos))

plt.gca().set_aspect('equal')
plt.gca().set_xlabel('x')
plt.gca().set_ylabel('y')
# In[]:
eig_vals, eig_vec = np.linalg.eig(c.cov) # linalg.eig:長さ1の固有ベクトルを返す

print("eig_vals:", eig_vals)
print("eig_vec:", eig_vec)
print("固有ベクトル1:", eig_vec[:,0])
print("固有ベクトル2:", eig_vec[:,1])


# In[]:
plt.contour(x, y, c.pdf(pos))

# 固有ベクトル1（長さ固有値１）の描画
v = 2*math.sqrt(eig_vals[0])*eig_vec[:,0]
plt.quiver(c.mean[0], c.mean[1], v[0], v[1], color="red", angles='xy', scale_units='xy', scale=1)

# 固有ベクトル2（長さ固有値2）の描画
v = 2*math.sqrt(eig_vals[1])*eig_vec[:,1]
plt.quiver(c.mean[0], c.mean[1], v[0], v[1], color="blue", angles='xy', scale_units='xy', scale=1)

plt.gca().set_aspect('equal')
plt.show()

# In[]:
V = eig_vec
L = np.diag(eig_vals)

print("分解したものを計算:\n", V.dot(L.dot(np.linalg.inv(V))))
print("下の共分散行列:\n", np.array([[100, -25*math.sqrt(3)], [-25*math.sqrt(3), 50]]))
