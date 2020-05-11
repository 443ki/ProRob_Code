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

# In[]:
p_t = pd.DataFrame(probs.sum())
p_t.plot()
p_t.transpose()

# In[]:
p_t.sum()

# In[]:
p_z = pd.DataFrame(probs.transpose().sum())
p_z.plot()
p_z.transpose()

# In[]:
p_z.sum()

# In[]:
cond_z_t = probs/p_t[0]
cond_z_t

# In[]:
(cond_z_t[6]).plot.bar(color="blue", alpha=0.5)
(cond_z_t[14]).plot.bar(color="orange", alpha=0.5)
plt.show()

# In[]:
# P(t|z)の計算
cond_t_z = probs.transpose()/probs.transpose().sum()

# センサ値が630になる確率
print("P(z=630) = ", p_z[0][630])
# 時間が13時の確率
print("P(t=13) = ", p_t[0][13])
# センサ値が630の時，時間が13時の条件付き確率
print("P(t=13 | z=630) = ", cond_t_z[630][13])
# ベイズの定理
print("Bsyes P(z=630 | t=13) = ", cond_t_z[630][13]*p_z[0][630]/p_t[0][13])

# 時間が13時に，センサ値が630になる条件付き確率
print("answer P(z=630 | t=13) = ", cond_z_t[13][630])

# In[]:
def bayes_estimation(sensor_value, current_estimation):
    new_estimation = []
    # それぞれの時間でP(z|t)P(t)を計算する
    for i in range(24):
        new_estimation.append(cond_z_t[i][sensor_value]*current_estimation[i])

    # 正規化
    return new_estimation/sum(new_estimation)

# In[]:
estimation = bayes_estimation(630, p_t[0])
plt.plot(estimation)

# In[]:
values_5 = [630,632,636]

estimation = p_t[0]

for v in values_5:
    estimation = bayes_estimation(v, estimation)

plt.plot(estimation)

# In[]:
values_11 = [617,624,619]

estimation = p_t[0]

for v in values_11:
    estimation = bayes_estimation(v, estimation)

plt.plot(estimation)
