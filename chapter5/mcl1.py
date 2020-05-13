import sys
sys.path.append('../scripts/')
from robot import *

# In[]:
class EstimationAgent(Agent):
    def __init__(self, nu, omega):
        super().__init__(nu, omega)

    def draw(self, ax, elems):
        elems.append(ax.text(0, 0, "hoge", fontsize=10))

# In[]:
world = World(30, 0.1)

### �n�}���쐬����3�����h�}�[�N��ǉ� ###
m = Map()
for ln in [(-4,2), (2,-3), (3,3)]:
    m.append_landmark(Landmark(*ln)) # *:���X�g�𕪉�����
world.append(m)

### ���{�b�g����� ###
initial_pose = np.array([2, 2, math.pi/6]).T
circling = EstimationAgent(0.2, 10.0/180*math.pi)
r = Robot(initial_pose, sensor=Camera(m), agent=circling)
world.append(r)

### �A�j���[�V�������s ###
world.draw()