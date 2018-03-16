# 隐藏模式， 自然有一些不同之处，要好好理解joint

# a world with obstacles --> a maze   有障碍物的迷宫世界， 改编自game_engine.py
# modified based on game_engine.py

from __future__ import print_function
import numpy as np
from game_maze import GameMaze
import pygame
from basic import *

inf = 100000

def check_x_joint(k, b, x0, x1, y0, y1, x):     # 检查x节点， 并且利用合适的范围内的x来计算y
    if x < x0 or x > x1:
        return False
    if k >= inf:
        return False
    y = k*x+b
    return y0 <= y <= y1

def check_y_joint(k, b, y0, y1, x0, x1, y):     # 检查y节点， 并且利用合适的范围内的y来计算x
    if y < y0 or y > y1:
        return False
    if np.abs(k) < 0.00001:     # 除数不能为0
        return False
    x = (y-b)/k
    return x0 <= x <= x1

def get_shoot_err(d):     # 获取涉及错误率， 不知道这个错误率怎么计算的
    sigma = d*100
    err = np.abs(np.random.randn() * sigma)
    return err

def fill_player_zone(state, center, size, value):  #  填充player的区域， 先计算四个边界的位置， 确定其形状， 区域， 然后填充上给定的值value。
    left, right = int(center[0] - size/2), int(center[0] + size/2)
    top, bottom = int(center[1] - size/2), int(center[1] + size/2)
    state[top:bottom, left:right] = value
    return state

class GameMazeHiding(GameMaze):
    """
    Hiding mode in maze world     迷宫世界中的隐藏模式
    """
    def __init__(self, training, screen_size, rabbit_size, gun_size, rabbit_speed, gun_speed,
                 history_size, max_time_step, debug, obstacle_size):
        """
        Initialization.
        :param screen_size: screen size of the game canvas
        :param rabbit_size: rabbit size
        :param gun_size: gun size
        :param rabbit_speed: speed of rabbit
        :param gun_speed: speed of gun
        :param with_gui: whether to display game canvas, useful when training in remote server. 是否显示游戏画布， 在远程服务器中训练时有用。
        :param debug: whether to print debugging info.
        """
        super(GameMazeHiding, self).__init__(training, screen_size, rabbit_size, gun_size,
                                             rabbit_speed, gun_speed, history_size,
                                             max_time_step, debug, obstacle_size)
        self.horizons = []     # 水平的
        self.verticles = []    # 垂直的

        for obs in self.obstacles:
            top, left, width, height = obs.top, obs.left, obs.width, obs.height   # 获取障碍物的四个边界
            self._add_obstacle_edges(left, top, width, height)   # 增加障碍物边界

    def _fill_obstacle_zone(self, value=0.3):                  # 填充障碍物区域
        for obs in self.obstacles:
            top, bottom = int(obs.top), int(obs.bottom)
            left, right = int(obs.left), int(obs.right)
            self.global_state[top:bottom, left:right] = value     # 确定四个边界之后， 围成一个区域， 这个区域内的网格都设定为一个相同的值value

    def _add_obstacle_edges(self, left, top, width, height):  # 增加障碍物的边界
        self.horizons.append((left, left+width, top))
        self.horizons.append((left, left+width, top+height))    #  为什么只有三个边呢？

        self.verticles.append((top, top+height, left))
        self.verticles.append((top, top+height, left+width))

    def _check_joint(self):    # 检查结合
        if self.rabbit_loc[0] == self.gun_loc[0]:  # 枪和兔子位置重合
            k = inf
            b = 0
        else:
            k = (self.rabbit_loc[1]-self.gun_loc[1]) / (self.rabbit_loc[0]-self.gun_loc[0])
            b = self.rabbit_loc[1] - k*self.rabbit_loc[0]                 # 计算直线的斜率和截距

        ymin = min(self.rabbit_loc[1], self.gun_loc[1])   # 兔子和枪的y值较小的那一个
        ymax = max(self.rabbit_loc[1], self.gun_loc[1])   # 兔子和枪的y值较大的那一个

        for x0, x1, y in self.horizons:    
            if check_y_joint(k, b, ymin, ymax, x0, x1, y):
                return True

        xmin = min(self.rabbit_loc[0], self.gun_loc[0])
        xmax = max(self.rabbit_loc[0], self.gun_loc[0])

        for y0, y1, x in self.verticles:
            if check_x_joint(k, b, xmin, xmax, y0, y1, x):
                return True
        return False                                          #  这个x和y是怎么检查的？    这里是用来判断隐藏hide的

    def reset(self):         # 隐藏模式重置
        self.hide = False
        return super(GameMazeHiding, self).reset()

    def get_reward_terminal(self):   # 环境反馈
        """
        Give the tuple (reward, terminal, other_info) to the agent  把元组给agent，
        :return: the tuple
        """
        flag = self._check_joint()        # check_joint 它怎么就能立flag呢？
        reward = 1 if flag else -1

        if self.rabbit_hit_boundary:      # 如果小兔子撞到边界， 就得惩罚
            reward += -1

        terminal = 1 if self.frame_cnt >= self.max_time_step else 0   # 帧数大于最大时间步时， 就终止了

        info = {'hide': flag}         # 信息： 就是隐藏flag

        self.text = '{}'.format(flag)    # 隐藏与否就是flag
        self.hide = flag

        return reward, terminal, info  # 反馈信息给agent

    def add_text_render(self, location=(0, 0)):
        """
        Add text to screen      # 在屏幕上添加文字信息， hide
        :return:
        """
        self.text = 'hide' if self.hide else ''
        super(GameMazeHiding, self).add_text_render()

    def render(self):
        """
        Render the canvas each step.
        :return:
        """
        if self.screen is None:
            self._init_screen()

        pygame.event.pump()
        self.screen.fill(BLACK)
        self._render_obstacles()
        self.add_text_render()
        # draw rabbit and gun
        color = WHITE if self.hide else RED      # 如果隐藏，颜色就是白色， 如果不隐藏， 就是红色。
        top_left = tuple(a-b for a, b in zip(self.rabbit_loc, (self.rabbit_size/2, self.rabbit_size/2)))
        pygame.draw.rect(self.screen, color, top_left + (self.rabbit_size, self.rabbit_size))
        top_left = tuple(a-b for a, b in zip(self.gun_loc, (self.gun_size/2, self.gun_size/2)))
        pygame.draw.rect(self.screen, GREEN, top_left + (self.gun_size, self.gun_size))
        if self.rabbit_action != 4:
            # draw action arrow
            offset = np.array([self.rabbit_size/2, 0])
            offset = self._rotate(ACTION2ANGLE[self.rabbit_action], offset)
            offset = offset + self.rabbit_loc
            pygame.draw.line(self.screen, GREEN, self.rabbit_loc, offset, 2)

        if self.gun_action != 4:
            offset = np.array([self.gun_size / 2, 0])
            offset = self._rotate(ACTION2ANGLE[self.gun_action], offset)
            offset = offset + self.gun_loc
            pygame.draw.line(self.screen, RED, self.gun_loc, offset, 2)

        pygame.display.update()
        FPSCLOCK.tick(FPS)          #  重新渲染画面
