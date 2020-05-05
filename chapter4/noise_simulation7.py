import sys
sys.path.append('../scripts/')
from ideal_robot import *
from scipy.stats import expon, norm, uniform     # exporn:指数分布，norm:ガウス分布， uniform:一様分布

# In[]:
class Robot(IdealRobot):    # クラスの継承

    def __init__(self, pose, agent=None, sensor=None, color="black", \
        noise_per_meter=5, noise_std=math.pi/60, \
        bias_rate_stds=(0.1,0.1), \
        expected_stuck_time = 1e100, expected_escape_time = 1e-100, \
        expected_kidnap_time=1e100, kidnap_range_x=(-5.0,5.0), kidnap_range_y=(-5.0,5.0)):
        super().__init__(pose, agent, sensor, color)    # IdeealRobotの__init__メソッドを呼び出す

        # 踏み石用確率密度関数の作成（指数分布）
        self.noise_pdf = expon(scale=1.0/(1e-100 + noise_per_meter))
        # 最初に小石を踏むまでの道のり
        self.distance_until_noise = self.noise_pdf.rvs()
        # thetaに加えるノイズ
        self.theta_noise = norm(scale=noise_std)

        # ロボット固有のバイアスの作成
        self.bias_rate_nu = norm.rvs(loc=1.0, scale=bias_rate_stds[0])
        self.bias_rate_omega = norm.rvs(loc=1.0, scale=bias_rate_stds[1])

        # スタック用確率密度関数の作成（指数分布）
        self.stuck_pdf = expon(scale=expected_stuck_time)
        self.escape_pdf = expon(scale=expected_escape_time)
        # 時間の初期化
        self.time_until_stuck = self.stuck_pdf.rvs()
        self.time_until_escape = self.escape_pdf.rvs()
        # ロボットがスタック中が表すグラフ
        self.is_stuck = False

        # 誘拐が起こる確率密度関数（指数分布）
        self.kidnap_pdf = expon(scale=expected_kidnap_time)
        self.time_until_kidnap = self.kidnap_pdf.rvs()
        # 誘拐後のロボット位置の範囲
        rx, ry = kidnap_range_x, kidnap_range_y
        # 誘拐後のロボットの位置・姿勢の確率密度関数（一様分布）
        self.kidnap_dist = uniform(loc=(rx[0], ry[0], 0.0), scale=(rx[1]-rx[0], ry[1]-ry[0], 2*math.pi))

    def noise(self, pose, nu, omega, time_interval):
        # 次に小石が踏むまでの道のりを経過時間の分減らす
        self.distance_until_noise -= abs(nu)*time_interval + self.r*abs(omega)*time_interval    # abs():絶対値
        # 小石を踏んだかの判定
        if self.distance_until_noise <= 0.0:
            # 次に小石を踏むまでの距離追加
            self.distance_until_noise += self.noise_pdf.rvs()
            # thetaにノイズ追加
            pose[2] += self.theta_noise.rvs()

        return pose

    def bias(self, nu, omega):
        return nu*self.bias_rate_nu, omega*self.bias_rate_omega

    def stuck(self, nu, omega, time_interval):
        # スタック中
        if self.is_stuck:
            self.time_until_escape -= time_interval
            # 脱出
            if self.time_until_escape <= 0.0:
                self.time_until_escape += self.escape_pdf.rvs()
                self.is_stuck = False

        # スタック中じゃない
        else:
            self.time_until_stuck -= time_interval
            if self.time_until_stuck <= 0.0:
                self.time_until_stuck += self.escape_pdf.rvs()
                self.is_stuck = True

        return nu*(not self.is_stuck), omega*(not self.is_stuck)

    def kidnap(self, pose, time_interval):
        self.time_until_kidnap -= time_interval
        if self.time_until_kidnap <= 0.0:
            self.time_until_kidnap += self.kidnap_pdf.rvs()
            return np.array(self.kidnap_dist.rvs()).T
        else:
            return pose

    def one_step(self, time_interval):
        if not self.agent:
            return
        obs = self.sensor.data(self.pose) if self.sensor else None
        nu, omega = self.agent.decision(obs)
        nu, omega = self.bias(nu, omega)
        nu, omega = self.stuck(nu, omega, time_interval)
        self.pose = self.state_transition(nu, omega, time_interval, self.pose)
        self.pose = self.noise(self.pose, nu, omega, time_interval)
        self.pose = self.kidnap(self.pose, time_interval)

# In[]:
class Camera(IdealCamera):
    def __init__(self, env_map,
            distance_range=(0.5, 6.0),
            direction_range=(-math.pi/3, math.pi/3),
            distance_noise_rate=0.1, direction_noise=math.pi/90):
        super().__init__(env_map, distance_range, direction_range)

        self.distance_noise_rate = distance_noise_rate
        self.direction_noise = direction_noise

    def noise(self, relpos):
        # ガウス分布でノイズ生成
        ell = norm.rvs(loc=relpos[0], scale=relpos[0]*self.distance_noise_rate)
        phi = norm.rvs(loc=relpos[1], scale=self.direction_noise)
        return np.array([ell, phi]).T

    def data(self, cam_pose):
        observed = []
        for lm in self.map.landmarks:
            z = self.observation_function(cam_pose, lm.pos)
            if self.visible(z):
                z = self.noise(z)
                observed.append((z, lm.id))

        self.lastdata = observed
        return observed

# In[]:
world = World(30, 0.1)

### 地図を生成してランドマークを3つ追加する ###
m = Map()
m.append_landmark(Landmark(-4, 2))
m.append_landmark(Landmark(2, -3))
m.append_landmark(Landmark(3, 3))
world.append(m)

### ロボットを作る ###
circling = Agent(0.2, 10.0/180*math.pi)
r = Robot( np.array([0, 0, 0]).T, sensor=Camera(m), agent=circling)
world.append(r)

### アニメーション実行 ###
world.draw()
