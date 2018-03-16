from __future__ import print_function
from game_engine import GameEngine
import numpy as np
import pygame

class GameKeepHitRate(GameEngine):
    """
    Game Simulator for keep hit rate mode. 保持命中率的模式的游戏模拟器， 而且命中率和距离也有关

    The goal of the rabbit is to keep certain distance with the gun 这段注释是错误的？
    """
    def __init__(self, training, screen_size, rabbit_size, gun_size, rabbit_speed, gun_speed,
                 history_size, max_time_step, debug, desired_goal):
        """
        Initialization, see `GameEngine.__init__()` for detail
        """
        # check input arg
        if desired_goal < 0. or desired_goal > 1.: 
            raise RuntimeError('desired hit rate goal should be in the rage [0, 1], while given {}'.   # 当然， 命中率必须得在0，1之间
                               format(desired_goal))

        self.distance = 0.            # 距离
        self.hit_rate = 0.            # 命中率
        self.goal = desired_goal      # 理想目标

        super(GameKeepHitRate, self).__init__(training, screen_size, rabbit_size, gun_size,
                                              rabbit_speed, gun_speed, history_size,
                                              max_time_step, debug)

    def reset(self):
        """
        Reset ebvironment. See `GameState.reset()` for detail.
        """
        self.hit_rate = 0.
        self.distance = 0.
        return super(GameKeepHitRate, self).reset()

    @staticmethod
    def get_shoot_err(d):    # 命中率是怎么计算的？
        sigma = d*100
        err = np.abs(np.random.randn() * sigma)
        return err

    def get_reward_terminal(self):   # 这个要好好看看，仔细研究研究？？？？
        """
        Overwrite base class method.
        :return: tuple of (reward, terminal, info)
        """
        self.distance = float(np.linalg.norm(self.rabbit_loc - self.gun_loc)) / self.screen_size
        shoot_err = self.get_shoot_err(self.distance)

        self.hit_rate = float(self.frame_cnt - 1) / self.frame_cnt * self.hit_rate   # 这个命中率是怎么计算的

        if shoot_err < self.rabbit_size/2:
            self.hit_rate += 1.0/self.frame_cnt

        reward = -np.abs(self.hit_rate - self.goal)

        if self.rabbit_hit_boundary:
            reward = -1

        terminal = 1 if self.frame_cnt >= self.max_time_step else 0

        info = {'distance': self.distance, 'hit_rate': self.hit_rate}

        return reward, terminal, info

    def add_text_render(self, location=(0, 0)):
        """
        overwrite `GameEngine.add_text_render()`
        """
        self.text = 'dis: {:.2f}, hit_rate: {:.2f}, goal: {:.2f}'.format(
            self.distance, self.hit_rate, self.goal)
        super(GameKeepHitRate, self).add_text_render()
