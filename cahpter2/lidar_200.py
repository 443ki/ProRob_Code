import pandas as pd
data = pd. read_csv("sensor_data_200.txt", delimiter=" ", header=None, names=("data","time","ir","lidar"))
data
# In[]:
print(data["lidar"][0:5])

# In[]:
import matplotlib.pyplot as plt
data["lidar"].hist(bins = max(data["lidar"]) - min(data["lidar"]), align = 'left')
plt.show()

# In[]:
mean1 = sum(data["lidar"].values)/len(data["lidar"].values)
mean2 = data["lidar"].mean()
print(mean1, mean2)

# In[]:
data["lidar"].hist(bins = max(data["lidar"]) - min(data["lidar"]), color = "orange", align='left')
plt.vlines(mean1, ymin=0, ymax=5000, color="red")
plt.show()

# In[]:
### 定義から計算 ###
zs = data["lidar"].values
mean = sum(zs)/len(zs)
diff_square = [ (z - mean)**2 for z in zs]

# 標本分散
sampling_var = sum(diff_square)/(len(zs))
# 不偏分散
unbiased_var = sum(diff_square)/(len(zs)-1)

print(sampling_var)
print(unbiased_var)

### Pandasを使用 ###
# 標本分散
pandas_sampling_var = data["lidar"].var(ddof=0) # Falseでは何故かエラー
# デフォルト（不偏分散）
pandas_default_var = data["lidar"].var()

print(pandas_sampling_var)
print(pandas_default_var)

### NumPayを使用 ###
import numpy as np

numpy_defalt_var = np.var(data["lidar"])
numpy_unbiased_var = np.var(data["lidar"], ddof=1)

print(numpy_defalt_var)
print(numpy_unbiased_var)

# In[]:
import math

### 定義から計算 ###
stddev1 = math.sqrt(sampling_var)
stddev2 = math.sqrt(unbiased_var)

### Pandasを使用 ###
pandas_stddev = data["lidar"].std()

print(stddev1)
print(stddev2)
print(pandas_stddev)

# In[]:
freqs = pd.DataFrame(data["lidar"].value_counts())
freqs.transpose()

# In[]:
freqs["probs"] = freqs["lidar"]/len(data["lidar"])
freqs.transpose()

# In[]:
sum(freqs["probs"])

# In[]:
freqs["probs"].sort_index().plot.bar(color="blue")
plot.show()

# In[]:
def drawing():
    return freqs.sample(n=1, weights="probs").index[0]

drawing()

# In[]:
#samples = [ drawing() for i in range(100)]
samples = [ drawing() for i in range(len(data))]
simulated = pd.DataFrame(samples, columns=["lidar"])
p = simulated["lidar"]
p.hist(bins = max(p) - min(p), color="orange", align='left')
plt.show()

# In[]:
def p(z, mu=209.7, dev=23.4):
    return math.exp(-(z - mu)**2/(2*dev))/math.sqrt(2*math.pi*dev)

# In[]:
zs = range(190,230)
ys = [p(z) for z in zs]

plt.plot(zs, ys)
plt.show()

# In[]:
def prob(z, width=0.5):
    return width*( p(z-width) + p(z+width) )

zs = range(190,230)
ys = [prob(z) for z in zs]

plt.bar(zs, ys, color="red", alpha=0.3)
f = freqs["probs"].sort_index()
plt.bar(f.index, f.values, color="blue", alpha=0.3)
plt.show()
