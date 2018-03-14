import numpy as np 
import pygame

# some constant
# some colors
RED    = (255, 0,   0  )
GREEN  = (0,   255, 0  )
BLUE   = (0,   0,   255)
BLACK  = (0,   0,   0  )
WHITE  = (255, 255, 255)
YELLOW = (255, 255, 0  )

# predefined colors
PREDEFINED_COLORS = {'red': RED, 
					'green': GREEN, 
					'blue': BLUE, 
					'black': BLACK, 
					'white': WHITE, 
					'yellow': YELLOW}


# game configuration
FPS = 30     # 画面每秒传输帧数， fps越大， 画面越流畅。
FPSCLOCK = pygame.time.Clock()

pygame.font.init()
Font = pygame.font.SysFont('Comic Sans MS', 30)

# position delta for each action
DELTA2D = [(dx, dy) for dx in xrange(-1, 2) for dy in xrange(-1, 2)]   x , y ∈ { -1, 0, 1 }


# action mapping to direction to move     动作映射成方向来移动  ， 真不知道师兄是怎么标注这些方向的。
ACTION2ANGLE = {0: -np.pi / 4 * 3,
                1: np.pi,
                2: np.pi / 4 * 3,
                3: -np.pi / 2,
                4: 0,
                5: np.pi / 2,
                6: -np.pi / 4,
                7: 0,
                8: np.pi / 4}

# action for DO NOTHING
DONOTHING_INDEX = DELTA2D.index((0, 0))

# position delta for each angle
# ANGLE2DELTA[i] = j: if you want to move towards to the angle[i], you should
# move delta[j]
ANGLES = [np.pi/4*i for i in range(8)]
ANGLE2DELTA = {0: 7, 1: 8, 2: 5, 3: 2, 4: 1, 5: 0, 6: 3, 7: 6}

# for compatible with openai gym  为了与openai gym兼容
class ActionSpace(object):      # 动作空间
    def __init__(self, actions):
        """
        Action Space.
        :param actions: action numbers   # 动作数字， 在这里，动作是与数字对应的， 方便计算机理解
        """
        self.actions = actions

    def sample(self):
        """
        Sample action randomly     #  对动作采样
        :return:
        """
        return np.random.randint(self.actions)


# obstacles in the maze world   迷宫世界的障碍物
class Obstacle(object):
    def __init__(self, left, top, width, height, transparency=0.):
        """
        Shape and position initialization of obstacle.   障碍物形状与位置的初始化
        :param top: the top coordinate of the obstacle (y coordinate)    上坐标 = y
        :param left: the left coordinate of the obstacle (x coordinate)  左坐标 = x
        :param width: the width     宽度
        :param height: the height   高度
        """
        self.top, self.left = top, left   
        self.bottom, self.right = top + height, self.left + width    # 坐标值得加上高和宽
        self.width, self.height = width, height 
        self.transparency = transparency    # 还包括透明度

    def contain_point(self, p):
        """
        Check if the point p is inside the obstacle     检查点p是否在障碍物里面
        :param p: the point to check. p(x, y)
        :return: boolean indicator
        """
        return self.left <= p[0] <= self.right and self.top <= p[1] <= self.bottom   # 如果坐标值在障碍物的范围内， 就代表在障碍物里面。


# paticipators   参与者
class Player(object):
    """ Plays in the game. Base class of `Rabbit` and `Gun`.   参与者， 就是兔子rabbit和枪gun的基类。
    """
    def __init__(self, size, speed, color):
        """ Initialization.

        :param size(int): size of the square standing for the player   代表参与者player的方块尺寸
        :param speed(int): moving speed of the player                  参与者的移动速度
        :param color(str/tuple of integers): color                     颜色（数字的字符串或者元组）
        """
        if isinstance(color, str):
            if color not in PREDEFINED_COLORS:
                raise RuntimeError('Unrecognized color name: {}, '
                                   'availables are {}'.format(color,
                                   ','.join(PREDEFINED_COLORS.keys())))
            self.color = PREDEFINED_COLORS[color]
        else:
            assert isinstance(color, tuple)
            self.color = color
        self.size = size
        self.speed = speed          # 参与者的基本属性， 颜色、尺寸、速度

# rabbit
class Rabbit(Player):
    """ Rabbit in the game.
    """
    def __init__(self, size, speed, color):
        """ Initialization.
        """
        super(Rabbit, self).__init__(size, speed, color)

# gun
class Gun(Player):
    """ Gun in the game
    """
    def __init__(self, size, speed, color):
        super(Gun, self).__init__(size, speed, color)
