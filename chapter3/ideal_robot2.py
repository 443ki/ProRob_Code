import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches
import numpy as np

# In[]:
class World:
    def __init__(self):
        #　ここにロボット等のオブジェクトを登録
        self.objects = []

    # オブジェクト登録のための関数
    def append(self, obj):
        self.objects.append(obj)

    def draw(self):
        # 8x8 inchの図を準備
        fig = plt.figure(figsize=(8,8))

        # サブプロットを準備
        ax = fig.add_subplot(111)

        # 縦横比を座標の値と一致させる
        ax.set_aspect('equal')

        # X軸, Y軸を-5m x 5mの範囲で描画
        ax.set_xlim(-5,5)
        ax.set_ylim(-5,5)

        # X軸, Y軸にラベルを表示する
        ax.set_xlabel("X", fontsize=20)
        ax.set_ylabel("Y", fontsize=20)

        # appendした物体を次々に描画
        for obj in self.objects:
            obj.draw(ax)

        plt.show()

# In[]:
class IdealRobot:
    def __init__(self, pose, color="black"):
        # 引数から姿勢の初期値を設定
        self.pose = pose

        # 描画のための固定値
        self.r = 0.2

        #引数から描画するときの色を設定
        self.color = color

    def draw(self, ax):
        # 姿勢の変数を分解
        x, y, theta = self.pose

        # ロボットの鼻先のx, y座標
        xn = x + self.r * math.cos(theta)
        yn = y + self.r * math.sin(theta)

        # ロボットの向きを示す線分の描画
        ax.plot([x,xn], [y,yn], color=self.color)

        # ロボットの胴体を示す円を作って，サブプロットに登録
        c = patches.Circle(xy=(x, y), radius=self.r, fill=False, color=self.color)
        ax.add_patch(c)

# In[]:
world = World()

# ロボットのインスタンスを生成(色省略)
robot1 = IdealRobot( np.array([2, 3, math.pi/6]).T )

# ロボットのインスタンスを生成(色指定)
robot2 = IdealRobot( np.array([-2, -1, math.pi/5*6]).T, "red" )

#ロボットを登録
world.append(robot1)
world.append(robot2)

world.draw()
