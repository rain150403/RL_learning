# a world with obstacles --> a maze 一个带有障碍物的场景， 迷宫， 基于 game_engine.py 改编
# modified based on game_engine.py

from __future__ import print_function
import pygame
from pygame.locals import *
import numpy as np
import math
from collections import deque
from basic import *
from game_engine import GameEngine


class GameMaze(GameEngine):
    """
    Game state manager for maze.   迷宫游戏状态管理员

    We will add square obstacles in the near-corners in the grid world.     在网格世界中靠近角落的位置放置方形障碍物。
    """
    def __init__(self, training, screen_size, rabbit_size, gun_size, rabbit_speed, gun_speed,
                 history_size, max_time_step, debug, obstacle_size):    # 从game engine继承而来， 只多了一个obstacle size。
        """
        Initialization.
        :param screen_size: screen size of the game canvas       -->  游戏画布中的屏幕尺寸
        :param rabbit_size: rabbit size                          -->  兔子尺寸
        :param gun_size: gun size                                -->  枪的尺寸
        :param rabbit_speed: speed of rabbit                     -->  兔子的速度
        :param gun_speed: speed of gun                           -->  枪的速度
        :param history_size: historical size                     -->  历史经历的规模
        :param debug: whether to print debugging info.           -->  是否打印debug信息
        :param obstacle_size: obstacle size                      -->  障碍物尺寸
        """
        # create the obstacles    创建障碍物
        if obstacle_size >= 0.5:      # 障碍物的尺寸为什么要小于0.5， 这里面的尺寸是什么样的比例， 比如整个画面有多大，兔子和枪在里面有多小， 障碍物和他们比，有是处于什么位置。
            raise RuntimeError('Obstacle size shoule be less than 0.5, while {} is given'.
                               format(obstacle_size))
        self.obstacle_size = obstacle_size
        self.screen_size = screen_size

        super(GameMaze, self).__init__(training, screen_size, rabbit_size, gun_size,
                                       rabbit_speed, gun_speed, history_size,
                                       max_time_step, debug)
        self._create_obstacles()

    def _create_obstacles(self):    # （静态）障碍物的创建
        self.obstacles = []
        obstacle_size = self.screen_size * self.obstacle_size    # oh， 明白了， 给的障碍物尺寸， 其实是占屏幕的比例， 所以得小于0.5
        width, height = obstacle_size, obstacle_size      # 高和宽都是这个尺寸， 自然是正方形

        left_size = self.screen_size * (1.0 - self.obstacle_size * 2)

        left, top = left_size / 3.0, left_size / 3.0
        self.obstacles.append(Obstacle(left, top, width, height))

        left += left_size / 3.0 + width
        self.obstacles.append(Obstacle(left, top, width, height))

        top += left_size / 3.0 + height
        self.obstacles.append(Obstacle(left, top, width, height))

        left -= left_size / 3.0 + width
        self.obstacles.append(Obstacle(left, top, width, height))    # 反正就是一种确定障碍物位置的方法， 需要的时候可以仔细研究

    def _render_obstacles(self):    # 渲染障碍物  ， 画出障碍物
        for obs in self.obstacles:
            top_left = obs.left, obs.top
            if obs.transparency < 0.2:
                pygame.draw.rect(self.screen, BLUE, top_left + (obs.width, obs.height))
            else:
                pygame.draw.rect(self.screen, YELLOW, top_left + (obs.width, obs.height))  # 这时候就可以画障碍物了

    def check_in_obstacle(self, p):   # 检查p点是否在障碍物中
        """
        Check a point `p` is in an obstacle
        :param p:
        :return:
        """
        for i, obs in enumerate(self.obstacles):
            if obs.contain_point(p):          # 确定在哪个障碍物中
                return i
        return -1

    def get_reward_terminal(self):
        """
        This is an example for keep distance in maze world.   在迷宫世界中保持距离的例子。
        就是获得反馈， 策略是让其保持距离， 做到了就给奖励， 帧数超过最大的时间步， 就结束一局， 信息自然就是距离啦，，返回这三个结果即可
        """
        self.distance = float(np.linalg.norm(self.rabbit_loc - self.gun_loc)) / self.screen_size
        reward = 1 if 0.3 <= self.distance <= 0.4 else -1
        terminal = 1 if self.frame_cnt >= self.max_time_step else 0

        info = {'distance': self.distance}

        return reward, terminal, info

    def add_text_render(self, location=(0, 0)):
        """
        Overwrite `GameEngine.add_text_render()`     加入文本信息的渲染， 这里会重写`GameEngine.add_text_render()`这个函数
        """
        super(GameMaze, self).add_text_render()

    def generate_loc_randomly(self):     # 生成随机位置，返回位置的np.ndarray数组
        """
        Generate locations randomly.
        :return: locations as `np.ndarray`
        """
        while True:
            rab_loc = np.random.rand(2) * self.screen_size
            gun_loc = np.random.rand(2) * self.screen_size
            if self.check_in_obstacle(rab_loc) == -1 and self.check_in_obstacle(gun_loc) == -1:   # 这里就是确保我们的agent不会出现在障碍物里面
                break     
        return rab_loc, gun_loc

    def update_loc(self, ori_loc, delta, speed):    # 更新位置， 给定原始位置、速度和增量， 就能计算出新的位置。 如果新的位置超出网格， 就返回原位置。
        """
        Update location by `delta*speed`. If the new location is beyond the grid, then return the original location.
        :param ori_loc: original location
        :param delta: location delta, given by `self.delta2d[action]`
        :param speed: a scalar to sclae `delta`
        :return: new location and a flag indicating whether hit the boundary
        """
        loc = ori_loc + np.array(delta) * speed
        if (loc < self.grid_size).all() and (loc >= 0).all() and self._check_crash(loc) == -1:
            return loc, False
        else:
            return ori_loc, True

    def render(self):
        """
        Render the canvas each step.    每一步都要渲染一下画布
        :return:
        """
        if self.screen is None:        
            self._init_screen()     # 初始化屏幕

        pygame.event.pump()
        self.screen.fill(BLACK)
        self._render_obstacles()
        self.add_text_render()
        # draw rabbit and gun
        top_left = tuple(a-b for a, b in zip(self.rabbit_loc, (self.rabbit_size/2, self.rabbit_size/2)))
        pygame.draw.rect(self.screen, RED, top_left + (self.rabbit_size, self.rabbit_size))
        top_left = tuple(a-b for a, b in zip(self.gun_loc, (self.gun_size/2, self.gun_size/2)))
        pygame.draw.rect(self.screen, GREEN, top_left + (self.gun_size, self.gun_size))
        if self.rabbit_action != DONOTHING_INDEX:
            # draw action arrow
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
        FPSCLOCK.tick(FPS)               # 就是重新填充整个屏幕， 显示字幕， 显示枪和兔子， 最后，显示整个画面

    def _check_crash(self, loc):
        """
        Check the rabbit or gun crash into the obstacle or not             检查枪或者兔子有没有撞到障碍物
        :return: the index of obstacle to crash. If not, return -1         返回障碍物的索引， 如果没有，就返回-1
        """
        for i, obs in enumerate(self.obstacles):
            if obs.contain_point(loc):                 # 如果这个位置信息包含在障碍物里， 就返回对应的索引， 没有，就返回-1
                return i
        return -1                         

if __name__ == '__main__':
     pygame.init()
     env = GameMaze(True, 500, 20, 20, 10, 10, 5, 200, True, 0.3)  # 这里的障碍物最大是0.5， 已经占据画面的2/3了， 平均每个占到快1/4了，当然，依然是数值越小就越小
     while True:
        state = env.reset()
        print (state)       # 最初的时候， 先输出一个状态
        env.render()
        terminal = False
        while not terminal:
            action = np.random.randint(4)
            state, reward, terminal, _ = env.step(action)
            print (state)     # 之后，在这一轮中， 每次都会输出当前状态  位置信息（4个数）的拼接是history， history（5个）拼接就是状态， 4*5 = 20， 显示正好有20个数
            env.render()


"""
(training, screen_size, rabbit_size, gun_size, rabbit_speed, gun_speed, history_size, max_time_step, debug)

training = True, screen_size = 500, rabbit_size = 20, gun_size = 20, rabbit_speed = 10, gun_speed = 10, history_size = 5, max_time_step = 200, debug = True
"""
