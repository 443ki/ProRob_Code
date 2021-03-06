import matplotlib.pyplot as plt

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
world = World()
world.draw()
