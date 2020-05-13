import sys
sys.path.append('../scripts/')
from robot import *

# In[]:
class Particle:
    def __init__(self, init_pose):
        self.pose = init_pose

# In[]:
class Mcl:
    def __init__(self, init_pose, num):
        self.particles = [Particle(init_pose) for i in range(num)]

    def draw(self, ax, elems):
        # �S�p�[�e�B�N���̍��W�����X�g��
        xs = [p.pose[0] for p in self.particles]
        ys = [p.pose[1] for p in self.particles]
        # �p�[�e�B�N���̌����̃x�N�g�����������X�g��
        vxs = [math.cos(p.pose[2]) for p in self.particles]
        vys = [math.sin(p.pose[2]) for p in self.particles]
        elems.append(ax.quiver(xs, ys, vxs, vys, color="blue", alpha=0.5))


# In[]:
class EstimationAgent(Agent):
    def __init__(self, nu, omega, estimator):
        super().__init__(nu, omega)
        self.estimator = estimator

    def draw(self, ax, elems):
        self.estimator.draw(ax, elems)

# In[]:
world = World(30, 0.1)

### �n�}���쐬����3�����h�}�[�N��ǉ� ###
m = Map()
for ln in [(-4,2), (2,-3), (3,3)]:
    m.append_landmark(Landmark(*ln)) # *:���X�g�𕪉�����
world.append(m)

### ���{�b�g����� ###
initial_pose = np.array([2, 2, math.pi/6]).T
estimator = Mcl(initial_pose, 100)
circling = EstimationAgent(0.2, 10.0/180*math.pi, estimator)
r = Robot(initial_pose, sensor=Camera(m), agent=circling)
world.append(r)

### �A�j���[�V�������s ###
world.draw()
