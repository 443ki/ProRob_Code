%matplotlib qt
import matplotlib.animation as anm
import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches
import numpy as np


# In[]:
class World:
    def __init__(self, debug=False):
        # ここにロボット等のオブジェクトを登録
        self.objects = []

        # デバッグ用フラグ
        self.debug = debug

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
        ax.set_xlabel("X", fontsize=10)
        ax.set_ylabel("Y", fontsize=10)

        # 描画する図形のリスト
        elems = []

        if self.debug:
            # デバッグ時はアニメーションさせない
            for i in range (1000):
                self.one_step(i, elms, ax)

        else:
            # アニメーションのためのオブジェクト作成
            self.ani = anm.FuncAnimation(fig, self.one_step, fargs=(elems, ax), frames=10, interval=1000, repeat=False)
            plt.show()

    # 1ステップ時刻を進める関数
    def one_step(self, i, elems, ax):
        while elems:
            # 二重描画を防ぐために図形をいったんクリア
            elems.pop().remove()

        # elemsにテキストオブジェクトを追加する
        elems.append(ax.text(-4.4, 4.5, "t="+str(i), fontsize=10))
        for obj in self.objects:
            obj.draw(ax, elems)

# In[]:
class IdealRobot:
    def __init__(self, pose, color="black"):
        # 引数から姿勢の初期値を設定
        self.pose = pose

        # 描画のための固定値
        self.r = 0.2

        #引数から描画するときの色を設定
        self.color = color

    def draw(self, ax, elems):
        # 姿勢の変数を分解
        x, y, theta = self.pose

        # ロボットの鼻先のx, y座標
        xn = x + self.r * math.cos(theta)
        yn = y + self.r * math.sin(theta)

        # ロボットの向きを示す線分の描画
        elems += ax.plot([x,xn], [y,yn], color=self.color)

        # ロボットの胴体を示す円を作って，サブプロットに登録
        c = patches.Circle(xy=(x, y), radius=self.r, fill=False, color=self.color)
        elems.append(ax.add_patch(c))

# In[]:
world = World(debug=False)

# ロボットのインスタンスを生成(色省略)
robot1 = IdealRobot( np.array([2, 3, math.pi/6]).T )

# ロボットのインスタンスを生成(色指定)
robot2 = IdealRobot( np.array([-2, -1, math.pi/5*6]).T, "red" )

#ロボットを登録
world.append(robot1)
world.append(robot2)

world.draw()
