from __future__ import print_function
import numpy as np
import pygame
import math
from collections import deque

from basic import *

class GameEngine(object):
    """
    Game state manager.     游戏状态管理员

    After receiving action from the agent, the manager updates the     
    game state and emits the tuple `(next_state, reward, terminal, info)`    
    to the agent. see `step()` for detail.
    从agent获取动作， 这个管理员就会更新游戏状态并产生这个元组（下一个状态， 奖励， 终止信号， 信息）给agent， 算是一种反馈。更详细的内容看函数step().
    """
    def __init__(self, training, screen_size,
                 rabbit_size, gun_size, rabbit_speed, gun_speed,
                 history_size, max_time_step, debug):
        """
        Initialization.  初始化
        :param training: training or testing                     训练还是预测？
        :param screen_size: screen size of the game canvas       游戏场景的屏幕尺寸
        :param rabbit_size: rabbit size                          兔子的尺寸
        :param gun_size: gun size                                枪的尺寸
        :param rabbit_speed: speed of rabbit                     兔子的速度
        :param gun_speed: speed of gun                           枪的速度
        :param history_size: historical size                     历史经历的规模
        :param debug: whether to print debugging info.           是否打印debug信息
        """
        self.training = training
        self.debug = debug
        self.screen_size = screen_size
        self.rabbit_size, self.gun_size = rabbit_size, gun_size

        self.grid_size = np.array([self.screen_size, self.screen_size])       # 屏幕尺寸只是一个数， 要变成数组才能成为网格grid的尺寸
        self.history = deque(maxlen=history_size)  
        # deque模块是python标准库collections中的一项，它提供了两端都可以操作的序列，这意味着，在序列的前后你都可以执行添加或删除操作。
        # 通俗地说， 把历史经历放在一个list或者队列里， 随时存取， 但是长度不超过history_size.

        self.rabbit_loc, self.gun_loc = None, None         # 兔子和枪的初始位置， 可随机产生， 也可自行指定
        self.rabbit_action = DONOTHING_INDEX           # 在basic.py里面提到过， 就是暂时什么都不做， 初始化
        self.gun_action = DONOTHING_INDEX

        self.max_time_step = max_time_step     # 最大时间步， 不知道什么作用？

        if self.max_time_step <= 0:
            raise ValueError('Invalid maximum time step: {}'.format(self.max_time_step))

        self.frame_cnt = 0         #  帧数  

        self.rabbit_speed, self.gun_speed = rabbit_speed, gun_speed

        self.action_space = ActionSpace(len(DELTA2D))     # 动作空间， 对动作采样

        # for rendering    用于渲染， 不知道是怎么个方法？
        self.screen = None
        self.text = ''

    def generate_loc_randomly(self):   # 随机产生位置
        """
        Generate locations randomly.
        :return: locations as `np.ndarray`   # 返回np.ndarray格式的位置。
        """
        # Create an array of the given shape and populate it with random samples from a uniform distribution over [0, 1).
		# (创建一个给定类型的数组，将其填充在一个均匀分布的随机样本[0, 1)中)
        # 简单来说， 在这里就是产生一个坐标， 且值都在0， 1之间， 就是为了产生在屏幕中的任意一个位置（坐标）。
        #
        rab_loc = (np.random.rand(2) * self.screen_size).astype(np.float32)
        gun_loc = (np.random.rand(2) * self.screen_size).astype(np.float32)
        return rab_loc, gun_loc

    def get_state(self):
        """
        Get state, invoked by `reset()` and `step()`   被函数reset() 和step()调用， 获取状态。
        :return: the state as 1d `np.ndarray`    以一维np.ndarray 的格式返回状态。
        """
        state = np.concatenate(tuple(self.history))    
        # 数组拼接， 把经历都拼接在一起， 形成我们的状态。  而且， 从后面内容可见， 所谓的history， 就是枪和兔子的位置信息的拼接， 且是0，1之间的数值。
        return state

    def _fill_history(self):
        locations = np.concatenate((self.gun_loc, self.rabbit_loc)).astype(np.float32) / self.screen_size   # 为什么要除以screen size？方便存储吗？
        self.history.append(locations)

    def reset(self):
        """
        Reset the environment.    重置环境 ，返回重置后的状态
        :return: the state after resetting
        """
        self.frame_cnt = 0
        self.rabbit_loc, self.gun_loc = self.generate_loc_randomly()   # 重置环境时，枪和兔子的位置是随机产生的

        self.history.clear()    # 清除历史经历， 仅仅是一个list，自带函数清除。
        for _ in range(self.history.maxlen):     # 重新填入经历作为history
            self._fill_history()

        return self.get_state()   # 有了history经历， 就有了状态

    def get_gun_action(self):
        """
        Give the gun action based on some strategy. Must be overwritten by sub-class     基于各种策略获取枪的动作， 必须被子类重写， 返回枪的动作。
        :return: gun action
        """
        if self.training:
            return self.action_space.sample()    # 如果是训练， 那就对动作空间采样
        else:
            key_states = pygame.key.get_pressed()
            if key_states[pygame.K_w]:
                # press 'w', the gun moves upward      上下两个动作与之前定义的动作数字刚好相反？ 而且在这里知道， 如果没有按给定的四个键， 那就默认向右走吗就是多余出来的4.
                gun_action = 3
            elif key_states[pygame.K_s]:
                # press 's', the gun moves downward
                gun_action = 5
            elif key_states[pygame.K_a]:
                # press 'a', the gun moves left-ward
                gun_action = 1
            elif key_states[pygame.K_d]:
                # press 'd', the gun moves right-ward
                gun_action = 7
            else:
                gun_action = 4
            return gun_action    # 如果是预测， 那就根据wsad方向键确定动作， 并最后返回相应的动作所对应的数字。

    def get_reward_terminal(self):  # # 我知道，这个函数其实没有具体实现， 要在它的子类里面自己实现。所以这个代码才不能运行

        """
        Give the tuple (reward, terminal, other_info) to the agent    把元组给agent， 不知道为什么没有操作， 还要产生一个错误信息
        :return: the tuple
        """
        raise NotImplementedError

    def set_text(self, text=''):    # 可能要设置一下输出的信息吧
        self.text = text

    def get_rabbit_action(self, action):
        """
        Get the rabbit action based on external input action (given by the agent)
        基于额外的动作输入（由agent给定）给出兔子的动作， 是不是可以这样理解， agent就是兔子，它做的动作， 自然就是兔子做的动作
        :return: rabbit action
        """
        return action

    """
    *** 这个函数很重要！！！  ***
    """

    def step(self, action):
        """
        Step forward after receiving action emitted by the agent.  收到由agent产生的动作后， 向前进
        Should overridden by all subclasses.     #  方法的覆盖(override)、重载(overload)和重写(overwrite)。。 这一方法应该被所有的子类覆盖
        :param action: action emitted by agent       由agent产生的动作
        :return: the `tuple(next_state, reward, terminal, info)`.    根据动作产生反馈， 而所谓的step， 应该就是有动作， 给出反馈的过程。
        """
        # first, increase the frame counter   首先， 画面帧数加一
        self.frame_cnt += 1
        # second, get the action and update the state   第二步， 获取动作， 更新状态
        self.gun_action = self.get_gun_action()
        self.rabbit_action = self.get_rabbit_action(action)   # 很好奇兔子的动作从哪来， 这个输入参数， 是一开始就给定的吗？

        self.rabbit_loc, self.rabbit_hit_boundary = self.update_loc(self.rabbit_loc,              # 做了动作就能有位置信息
                                                                    DELTA2D[self.rabbit_action],
                                                                    self.rabbit_speed)
        self.gun_loc, self.gun_hit_boundary = self.update_loc(self.gun_loc,
                                                              DELTA2D[self.gun_action],
                                                              self.gun_speed)
        self._fill_history()       # 位置信息拼接就是history

        # third, get the state of t+1
        next_state = self.get_state()    # history拼接就是状态， 也就是第三步， 获取t+1时刻的状态

        # forth, get tuple of (reward, terminal, info)     # 第四步， 获取信息元组  （然而， 这个函数里，并没有给出如何获得这个元组呀？）
        reward, terminal, info = self.get_reward_terminal()

        return next_state, reward, terminal, info  # 最后返回环境的反馈即可

    def get_state_dim(self):
        """
        Get state dimensions.        获取状态维度， 后面神经网络会用到
        :return:  state dimensions
        """
        return self.history.maxlen * 4    # 如何计算， 难道是4个坐标值？

    @staticmethod
    def get_action_dim():
        """
        Get number of available actions.       可选择的动作的数量， 也是后面神经网络会用到的
        :return: action space dimensions
        """
        return len(DELTA2D)                # 为何要这样计算

    def add_text_render(self, location=(0, 0)):
        """
        Add text to screen   在屏幕左上角添加文字， 一种添加文字的方式， 不用太在意， 照搬即可
        :return:
        """
        text_surface = Font.render(self.text, False, RED)
        self.screen.blit(text_surface, location)

    @staticmethod
    def _rotate(theta, p):    # 坐标旋转做什么用？
        cos, sin = np.cos(theta), np.sin(theta)
        ret_point = (cos*p[0]-sin*p[1], sin*p[0]+cos*p[1])
        return ret_point
    """
    坐标旋转公式:
    
    该公式仅仅针对旋转中心在坐标原点的情况。

	x1=cos(angle)*x-sin(angle)*y;
	y1=cos(angle)*y+sin(angle)*x;
	//angle为旋转的角度,x、y是旋转前的坐标
    """

    def _init_screen(self, caption='Gun Rabbit Game'):    # 屏幕初始化， 字幕caption = “枪打兔子游戏”， 设置字幕
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
        pygame.display.set_caption(caption)

    def render(self):   # 渲染
        """
        Rendering the game canvas.   渲染游戏场景（画布）
        :param other_render_func: rendering function for other specific component. See `GameMaze.render()` for example   
        其它特定部分的渲染函数，GameMaze.render()是一个例子 
        :return: None

        这一部分就是画画画， 初始化屏幕， 写字幕， 填充颜色，画上兔子和枪， 画上动作箭头， 最后显示。
        """
        if self.screen is None:
            self._init_screen()
        pygame.event.pump()
        # fill the screen with BLACK
        self.screen.fill(BLACK)
        # add text on the screen
        self.add_text_render()
        # draw rabbit and gun
        top_left = tuple(a-b for a, b in zip(self.rabbit_loc, (self.rabbit_size/2, self.rabbit_size/2)))
        pygame.draw.rect(self.screen, RED, top_left + (self.rabbit_size, self.rabbit_size))
        top_left = tuple(a-b for a, b in zip(self.gun_loc, (self.gun_size/2, self.gun_size/2)))
        pygame.draw.rect(self.screen, GREEN, top_left + (self.gun_size, self.gun_size))
        # draw action arrow
        if self.rabbit_action != DONOTHING_INDEX:
            offset = np.array([self.rabbit_size/2, 0])
            offset = self._rotate(ACTION2ANGLE[self.rabbit_action], offset)
            offset = offset + self.rabbit_loc
            pygame.draw.line(self.screen, GREEN, self.rabbit_loc, offset, 2)

        if self.gun_action != DONOTHING_INDEX:
            offset = np.array([self.gun_size / 2, 0])
            offset = self._rotate(ACTION2ANGLE[self.gun_action], offset)
            offset = offset + self.gun_loc
            pygame.draw.line(self.screen, RED, self.gun_loc, offset, 2)

        pygame.display.update()
        FPSCLOCK.tick(FPS)

    def update_loc(self, ori_loc, delta, speed):
        """
        用δ * 速度，来更新位置， 如果新的位置超过网格， 就返回最初的位置。
        Update location by `delta*speed`. If the new location is beyond the grid, then return the original location.
        :param ori_loc: original location        最初的位置
        :param delta: location delta, given by `DELTA2D[action]`      位置增量
        :param speed: a scalar to sclae `delta`  速度， 和 delte有一定的关系
        :return: new location and a flag indicating whether hit the boundary    新的位置，以及一个旗帜flag用于指示是否击中边界
        """
        loc = ori_loc + np.array(delta) * speed
        if (loc < self.grid_size).all() and (loc >= 0).all():
            return loc, False
        else:
            return ori_loc, True

    def generate_action_optimally(self):
        """
        Get optimal action: gun moves towards the rabbit directly    最优动作：所谓最优动作就是， 枪直接朝着兔子移动
        :return: the optimal action   返回最优动作
        """
        theta = self.get_theta()
        if theta < 0:
            theta += np.pi * 2
        k = int(np.ceil((theta-np.pi/8)/(np.pi/4)))      # 向上取整， 可能这个减去1/8是一个小技巧。
        if k == self.get_action_dim()-1:    # 超过动作个数， 就归零， 从零开始
            k = 0
        if self.debug:
            print('theta = {:.1f} degree, k = {}'.format(theta / np.pi * 180,     k))    # 如果设置debug信息， 那就打印这些信息， 角度theta对应的动作数字k
        assert k >= 0
        return ANGLE2DELTA[k]  #  角度和delta的对应      不知道这个和最优动作有什么关系

    def get_theta(self):
        """
        Get theta of the line connecting gun and agent.   连接枪和agent的线的theta角
        :return: the angle    返回角度   
        """
        d = self.rabbit_loc - self.gun_loc    
        theta = math.atan2(d[1], d[0])     # 根据兔子和枪的位置坐标， 通过数学方法计算theta值
        return theta

    def replay(self, coordinate_list, info=None):     # replay 重播、重放    # 坐标列表
        for i, coordinate in enumerate(coordinate_list):
            print('frame cnt: {}'.format(i))
            self.gun_loc = np.array(coordinate[:2])
            self.rabbit_loc = np.array(coordinate[2:])
            distance = np.linalg.norm(self.gun_loc-self.rabbit_loc)   # 求范数， 默认是二范数，也就是求距离   ，，这个距离是用在哪里的呢？

            pygame.event.pump()
            self.screen.fill(BLACK)
            if info is not None:
                self.text = info
            self.add_text_render()

            top_left = tuple(a - b for a, b in zip(self.rabbit_loc, (self.rabbit_size / 2, self.rabbit_size / 2)))
            pygame.draw.rect(self.screen, RED, top_left + (self.rabbit_size, self.rabbit_size))
            top_left = tuple(a - b for a, b in zip(self.gun_loc, (self.gun_size / 2, self.gun_size / 2)))
            pygame.draw.rect(self.screen, GREEN, top_left + (self.gun_size, self.gun_size))
            pygame.display.update()
            FPSCLOCK.tick(FPS)             # 还是画面填充与图形重画，并且重新展示画面
