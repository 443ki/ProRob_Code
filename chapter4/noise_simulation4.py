import sys
sys.path.append('../scripts/')
from ideal_robot import *
from scipy.stats import expon, norm     # exporn:指数分布，norm:ガウス分布

# In[]:
class Robot(IdealRobot):    # クラスの継承

    def __init__(self, pose, agent=None, sensor=None, color="black", \
        noise_per_meter=5, noise_std=math.pi/60, \
        bias_rate_stds=(0.1,0.1), \
        expected_stuck_time = 1e100, expected_escape_time = 1e-100):
        super().__init__(pose, agent, sensor, color)    # IdeealRobotの__init__メソッドを呼び出す

        # 指数分布のオブジェクト
        self.noise_pdf = expon(scale=1.0/(1e-100 + noise_per_meter))
        # 最初に小石を踏むまでの道のり
        self.distance_until_noise = self.noise_pdf.rvs()
        # thetaに加えるノイズ
        self.theta_noise = norm(scale=noise_std)

        # ロボット固有のバイアス
        self.bias_rate_nu = norm.rvs(loc=1.0, scale=bias_rate_stds[0])
        self.bias_rate_omega = norm.rvs(loc=1.0, scale=bias_rate_stds[1])

        # 確率密度関数の作成
        self.stuck_pdf = expon(scale=expected_stuck_time)
        self.escape_pdf = expon(scale=expected_escape_time)
        # 時間の初期化
        self.time_until_stuck = self.stuck_pdf.rvs()
        self.time_until_escape = self.escape_pdf.rvs()
        # ロボットがスタック中が表すグラフ
        self.is_stuck = False

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



    def one_step(self, time_interval):
        if not self.agent:
            return
        obs = self.sensor.data(self.pose) if self.sensor else None
        nu, omega = self.agent.decision(obs)
        nu, omega = self.bias(nu, omega)
        nu, omega = self.stuck(nu, omega, time_interval)
        self.pose = self.state_transition(nu, omega, time_interval, self.pose)
        self.pose = self.noise(self.pose, nu, omega, time_interval)

# In[]:
world = World(30, 0.1)

circling = Agent(0.2, 10.0/180*math.pi)

world = World(30, 0.1)

for i in range(100):
    circling = Agent(0.2, 10.0/180*math.pi)
    r = Robot(np.array([0, 0, 0]).T, sensor=None, agent=circling, color="gray", \
    noise_per_meter=0, bias_rate_stds=(0.0, 0.0), \
    expected_stuck_time=60.0, expected_escape_time=60.0)
    world.append(r)

r = IdealRobot( np.array([0, 0, 0]).T, sensor=None, agent=circling, color="red" )
world.append(r)

world.draw()
