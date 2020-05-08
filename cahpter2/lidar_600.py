import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("sensor_data_600.txt", delimiter=" ", header=None, names=("data","time","ir","lidar"))

data["lidar"].hist(bins = max(data["lidar"]) - min(data["lidar"]), align='left')
plt.show()

# In[]:
data.lidar.plot()
plt.show()

# In[]:
data["hour"] = [e//10000 for e in data.time]
d = data.groupby("hour")
d.lidar.mean().plot()
plt.show()

# In[]:
d.lidar.get_group(6).hist()
d.lidar.get_group(14).hist()
plt.show()

# In[]:
each_hour = { i : d.lidar.get_group(i).value_counts().sort_index() for i in range(24)}
freqs = pd.concat(each_hour, axis=1)
freqs = freqs.fillna(0)
probs = freqs/len(data)

probs

# in[]:
import seaborn as sns
sns.heatmap(probs)
plt.show()

# In[]:
sns.jointplot(data["hour"], data["lidar"], data, kind="kde")
plt.show()
