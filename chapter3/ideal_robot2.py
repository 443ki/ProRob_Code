import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches
import numpy as np

# In[]:
class World:
    def __init__(self):
        #�@�����Ƀ��{�b�g���̃I�u�W�F�N�g��o�^
        self.objects = []

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
        ax.set_xlabel("X", fontsize=20)
        ax.set_ylabel("Y", fontsize=20)

        # append�������̂����X�ɕ`��
        for obj in self.objects:
            obj.draw(ax)

        plt.show()

# In[]:
class IdealRobot:
    def __init__(self, pose, color="black"):
        # ��������p���̏����l��ݒ�
        self.pose = pose

        # �`��̂��߂̌Œ�l
        self.r = 0.2

        #��������`�悷��Ƃ��̐F��ݒ�
        self.color = color

    def draw(self, ax):
        # �p���̕ϐ��𕪉�
        x, y, theta = self.pose

        # ���{�b�g�̕@���x, y���W
        xn = x + self.r * math.cos(theta)
        yn = y + self.r * math.sin(theta)

        # ���{�b�g�̌��������������̕`��
        ax.plot([x,xn], [y,yn], color=self.color)

        # ���{�b�g�̓��̂������~������āC�T�u�v���b�g�ɓo�^
        c = patches.Circle(xy=(x, y), radius=self.r, fill=False, color=self.color)
        ax.add_patch(c)

# In[]:
world = World()

# ���{�b�g�̃C���X�^���X�𐶐�(�F�ȗ�)
robot1 = IdealRobot( np.array([2, 3, math.pi/6]).T )

# ���{�b�g�̃C���X�^���X�𐶐�(�F�w��)
robot2 = IdealRobot( np.array([-2, -1, math.pi/5*6]).T, "red" )

#���{�b�g��o�^
world.append(robot1)
world.append(robot2)

world.draw()
