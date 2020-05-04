%matplotlib qt
import matplotlib.animation as anm
import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches
import numpy as np

# In[]:
class World:
    def __init__(self, time_span, time_interval, debug=False):
        # ここにロボット等のオブジェクトを登録
        self.objects = []
        # デバッグ用フラグ
        self.debug = debug
        # シミュレーション時間
        self.time_span = time_span
        # 時間間隔
        self.time_interval = time_interval

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
            self.ani = anm.FuncAnimation(fig, self.one_step, fargs=(elems, ax), frames=int(self.time_span/self.time_interval)+1, interval=int(self.time_interval*1000), repeat=False)
            plt.show()

    # 1ステップ時刻を進める関数
    def one_step(self, i, elems, ax):
        while elems:
            # 二重描画を防ぐために図形をいったんクリア
            elems.pop().remove()

        # 時刻として表示する文字列
        time_str = "t=%.2f[s]" % (self.time_interval*i)
        # elemsにテキストオブジェクトを追加する
        elems.append(ax.text(-4.4, 4.5, time_str, fontsize=10))

        for obj in self.objects:
            obj.draw(ax, elems)
            # オブジェクトに"one_step"という名のメソッドがあれば実行
            if hasattr(obj, "one_step"):
                obj.one_step(self.time_interval)

# In[]:
class IdealRobot:
    def __init__(self, pose, agent=None, sensor=None, color="black"):
        # 引数から姿勢の初期値を設定
        self.pose = pose
        # 描画のための固定値
        self.r = 0.2
        # 引数から描画するときの色を設定
        self.color = color
        # エージェントを指定
        self.agent = agent
        # 軌跡の描画用
        self.poses = [pose]
        self.sensor = sensor

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
        # 軌跡の描画
        self.poses.append(self.pose)
        elems += ax.plot([e[0] for e in self.poses], [e[1] for e in self.poses], linewidth=0.5, color="black")
        if self.sensor and len(self.poses) > 1:
            self.sensor.draw(ax, elems, self.poses[-2])

    @classmethod    # クラスメソッド(インスタンス化しなくても実行可能なメソッド)
    def state_transition(cls, nu, omega, time, pose):
        t0 = pose[2]
        # 角速度がほぼ0の場合とそうでない場合で場合分け
        if math.fabs(omega)  < 1e-10:
            return pose + np.array([
                nu*math.cos(t0),
                nu*math.sin(t0),
                omega ]) * time
        else:
            return pose + np.array([
                nu/omega*(math.sin(t0 + omega*time) - math.sin(t0)),
                nu/omega*(-math.cos(t0 + omega*time) + math.cos(t0)),
                omega*time ])

    def one_step(self, time_interval):
        if not self.agent:
            return
        obs = self.sensor.data(self.pose) if self.sensor else None
        nu, omega = self.agent.decision(obs)
        self.pose = self.state_transition(nu, omega, time_interval, self.pose)

# In[]:
class Agent:
    def __init__(self, nu, omega):
        self.nu = nu
        self.omega = omega

    def decision(self, observation=None):
        return self.nu, self.omega

# In[]:
class Landmark:
    def __init__(self, x, y):
        self.pos = np.array([x, y]).T
        self.id = None

    def draw(self, ax, elems):
        # 散布図に点を打つためのメソッド
        c = ax.scatter(self.pos[0], self.pos[1], s=100, marker="*", label="landmarks", color="orange")
        elems.append(c)
        elems.append(ax.text(self.pos[0], self.pos[1], "id:" + str(self.id), fontsize=10))

# In[]:
class Map:
    def __init__(self):
        # 空のランドマークのリストを準備
        self.landmarks = []

    # ランドマークを追加する
    def append_landmark(self, landmark):
        # 追加するランドマークにIDを与える
        landmark.id = len(self.landmarks)
        self.landmarks.append(landmark)

    # 描画(Landmarkのdrawを順に呼び出す)
    def draw(self, ax, elems):
        for lm in self.landmarks:
            lm.draw(ax, elems)

# In[]:
class IdealCamera:
    def __init__(self, env_map, \
            distance_range=(0.5,6.0),
            direction_range=(-math.pi/3,math.pi/3)):
        self.map = env_map
        self.lastdata = []

        self.distance_range = distance_range
        self.direction_range = direction_range

    def visible(self, polarpos):
        if polarpos is None:
            return False

        return self.distance_range[0] <= polarpos[0] <= self.distance_range[1] and \
            self.direction_range[0] <= polarpos[1] <= self.direction_range[1]

    def data(self, cam_pose):
        observed = []
        for lm in self.map.landmarks:
            z = self.observation_function(cam_pose, lm.pos)
            if self.visible(z):
                observed.append((z, lm.id))

        self.lastdata = observed
        return observed

    @classmethod
    def observation_function(cls, cam_pose, obj_pos):
        diff = obj_pos - cam_pose[0:2] # [0:2]：スライス
        phi = math.atan2(diff[1], diff[0]) -cam_pose[2]
        while phi >= np.pi: phi -= 2*np.pi
        while phi < -np.pi: phi += 2*np.pi
        return np.array( [np.hypot(*diff), phi ] ).T

    def draw(self, ax, elems, cam_pose):
        for lm in self.lastdata:
            x, y, theta = cam_pose
            distance, direction = lm[0][0], lm[0][1]
            lx = x + distance * math.cos(direction + theta)
            ly = y + distance * math.sin(direction + theta)
            elems += ax.plot([x, lx], [y, ly], color='pink')

# In[]:
world = World(10, 0.1, debug=False)

# 地図を生成してランドマークを3つ追加する
m = Map()
m.append_landmark(Landmark(2, -2))
m.append_landmark(Landmark(-1, -3))
m.append_landmark(Landmark(3, 3))
#　worldに地図を登録
world.append(m)

# 0.2[m/s]で直進
straight = Agent(0.2, 0.0)
# 0.2[m/s], 10[deg/s]（円を描く）
circling = Agent(0.2, 10.0/180*math.pi)
# ロボットのインスタンスを生成（色省略,直進）
robot1 = IdealRobot( np.array([2, 3, math.pi/6]).T, sensor=IdealCamera(m), agent=straight )
# ロボットのインスタンスを生成（色指定，円）
robot2 = IdealRobot( np.array([-2, -1, math.pi/5*6]).T, sensor=IdealCamera(m), agent=circling, color="red")
#ロボットを登録
world.append(robot1)
world.append(robot2)

# アニメーション実行
world.draw()
