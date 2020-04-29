%matplotlib qt
import matplotlib.animation as anm
import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches
import numpy as np


# In[]:
class World:
    def __init__(self, debug=False):
        # �����Ƀ��{�b�g���̃I�u�W�F�N�g��o�^
        self.objects = []

        # �f�o�b�O�p�t���O
        self.debug = debug

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
            self.ani = anm.FuncAnimation(fig, self.one_step, fargs=(elems, ax), frames=10, interval=1000, repeat=False)
            plt.show()

    # 1�X�e�b�v������i�߂�֐�
    def one_step(self, i, elems, ax):
        while elems:
            # ��d�`���h�����߂ɐ}�`����������N���A
            elems.pop().remove()

        # elems�Ƀe�L�X�g�I�u�W�F�N�g��ǉ�����
        elems.append(ax.text(-4.4, 4.5, "t="+str(i), fontsize=10))
        for obj in self.objects:
            obj.draw(ax, elems)

# In[]:
class IdealRobot:
    def __init__(self, pose, color="black"):
        # ��������p���̏����l��ݒ�
        self.pose = pose

        # �`��̂��߂̌Œ�l
        self.r = 0.2

        #��������`�悷��Ƃ��̐F��ݒ�
        self.color = color

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

# In[]:
world = World(debug=False)

# ���{�b�g�̃C���X�^���X�𐶐�(�F�ȗ�)
robot1 = IdealRobot( np.array([2, 3, math.pi/6]).T )

# ���{�b�g�̃C���X�^���X�𐶐�(�F�w��)
robot2 = IdealRobot( np.array([-2, -1, math.pi/5*6]).T, "red" )

#���{�b�g��o�^
world.append(robot1)
world.append(robot2)

world.draw()
