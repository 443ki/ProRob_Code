%matplotlib qt
import matplotlib.animation as anm
import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches
import numpy as np

# In[]:
class World:
    def __init__(self, time_span, time_interval, debug=False):
        # �����Ƀ��{�b�g���̃I�u�W�F�N�g��o�^
        self.objects = []
        # �f�o�b�O�p�t���O
        self.debug = debug
        # �V�~�����[�V��������
        self.time_span = time_span
        # ���ԊԊu
        self.time_interval = time_interval

    # �I�u�W�F�N�g�o�^�̂��߂̊֐�
    def append(self, obj):
        self.objects.append(obj)

    def draw(self):
        # 8x8 inch�̐}������
        fig = plt.figure(figsize=(8,8))
        # �T�u�v���b�g������
        ax = fig.add_subplot(111)
        # �c��������W�̒l�ƈ�v������
        ax.set_aspect('equal')
        # X��, Y����-5m x 5m�͈̔͂ŕ`��
        ax.set_xlim(-5,5)
        ax.set_ylim(-5,5)
        # X��, Y���Ƀ��x����\������
        ax.set_xlabel("X", fontsize=10)
        ax.set_ylabel("Y", fontsize=10)
        # �`�悷��}�`�̃��X�g
        elems = []

        if self.debug:
            # �f�o�b�O���̓A�j���[�V���������Ȃ�
            for i in range (1000):
                self.one_step(i, elms, ax)
        else:
            # �A�j���[�V�����̂��߂̃I�u�W�F�N�g�쐬
            self.ani = anm.FuncAnimation(fig, self.one_step, fargs=(elems, ax), frames=int(self.time_span/self.time_interval)+1, interval=int(self.time_interval*1000), repeat=False)
            plt.show()

    # 1�X�e�b�v������i�߂�֐�
    def one_step(self, i, elems, ax):
        while elems:
            # ��d�`���h�����߂ɐ}�`����������N���A
            elems.pop().remove()

        # �����Ƃ��ĕ\�����镶����
        time_str = "t=%.2f[s]" % (self.time_interval*i)
        # elems�Ƀe�L�X�g�I�u�W�F�N�g��ǉ�����
        elems.append(ax.text(-4.4, 4.5, time_str, fontsize=10))

        for obj in self.objects:
            obj.draw(ax, elems)
            # �I�u�W�F�N�g��"one_step"�Ƃ������̃��\�b�h������Ύ��s
            if hasattr(obj, "one_step"):
                obj.one_step(self.time_interval)

# In[]:
class IdealRobot:
    def __init__(self, pose, agent=None, color="black"):
        # ��������p���̏����l��ݒ�
        self.pose = pose
        # �`��̂��߂̌Œ�l
        self.r = 0.2
        # ��������`�悷��Ƃ��̐F��ݒ�
        self.color = color
        # �G�[�W�F���g���w��
        self.agent = agent
        # �O�Ղ̕`��p
        self.poses = [pose]

    def draw(self, ax, elems):
        # �p���̕ϐ��𕪉�
        x, y, theta = self.pose
        # ���{�b�g�̕@���x, y���W
        xn = x + self.r * math.cos(theta)
        yn = y + self.r * math.sin(theta)
        # ���{�b�g�̌��������������̕`��
        elems += ax.plot([x,xn], [y,yn], color=self.color)
        # ���{�b�g�̓��̂������~������āC�T�u�v���b�g�ɓo�^
        c = patches.Circle(xy=(x, y), radius=self.r, fill=False, color=self.color)
        elems.append(ax.add_patch(c))
        # �O�Ղ̕`��
        self.poses.append(self.pose)
        elems += ax.plot([e[0] for e in self.poses], [e[1] for e in self.poses], linewidth=0.5, color="black")
        print(self.poses)

    @classmethod    # �N���X���\�b�h(�C���X�^���X�����Ȃ��Ă����s�\�ȃ��\�b�h)
    def state_transition(cls, nu, omega, time, pose):
        t0 = pose[2]
        # �p���x���ق�0�̏ꍇ�Ƃ����łȂ��ꍇ�ŏꍇ����
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
        nu, omega = self.agent.decision()
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
        # �U�z�}�ɓ_��ł��߂̃��\�b�h
        c = ax.scatter(self.pos[0], self.pos[1], s=100, marker="*", label="landmarks", color="orange")
        elems.append(c)
        elems.append(ax.text(self.pos[0], self.pos[1], "id:" + str(self.id), fontsize=10))

# In[]:
class Map:
    def __init__(self):
        # ��̃����h�}�[�N�̃��X�g������
        self.landmarks = []

    # �����h�}�[�N��ǉ�����
    def append_landmark(self, landmark):
        # �ǉ����郉���h�}�[�N��ID��^����
        landmark.id = len(self.landmarks)
        self.landmarks.append(landmark)

    # �`��(Landmark��draw�����ɌĂяo��)
    def draw(self, ax, elems):
        for lm in self.landmarks:
            lm.draw(ax, elems)

# In[]:
class IdealCamera:
    def __init__(self, env_map):
        self.map = env_map

    def data(self, cam_pose):
        observed = []
        for lm in self.map.landmarks:
            p = self.observation_function(cam_pose, lm.pos)
            observed.append((p, lm.id))

        return observed

    @classmethod
    def observation_function(cls, cam_pose, obj_pos):
        diff = obj_pos - cam_pose[0:2] # [0:2]�F�X���C�X
        phi = math.atan2(diff[1], diff[0]) -cam_pose[2]
        while phi >= np.pi: phi -= 2*np.pi
        while phi < -np.pi: phi += 2*np.pi
        return np.array( [np.hypot(*diff), phi ] ).T

# In[]:
world = World(10, 0.1, debug=False)

# �n�}�𐶐����ă����h�}�[�N��3�ǉ�����
m = Map()
m.append_landmark(Landmark(2, -2))
m.append_landmark(Landmark(-1, -3))
m.append_landmark(Landmark(3, 3))
#�@world�ɒn�}��o�^
world.append(m)

# 0.2[m/s]�Œ��i
straight = Agent(0.2, 0.0)
# 0.2[m/s], 10[deg/s]�i�~��`���j
circling = Agent(0.2, 10.0/180*math.pi)
# ���{�b�g�̃C���X�^���X�𐶐��i�F�ȗ�,���i�j
robot1 = IdealRobot( np.array([2, 3, math.pi/6]).T, straight )
# ���{�b�g�̃C���X�^���X�𐶐��i�F�w��C�~�j
robot2 = IdealRobot( np.array([-2, -1, math.pi/5*6]).T, circling, "red")
# ���{�b�g�̃C���X�^���X�𐶐��i�G�[�W�F���g�^���Ȃ��j
robot3 = IdealRobot( np.array([0, 0, 0]).T, color="blue")
#���{�b�g��o�^
world.append(robot1)
world.append(robot2)
world.append(robot3)

# �A�j���[�V�������s
world.draw()

# In[]:
cam = IdealCamera(m)
p = cam.data(robot2.pose)
print(p)
