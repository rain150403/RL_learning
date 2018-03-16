from __future__ import print_function
from game_engine import GameEngine
import numpy as np
import pygame

class GameKeepDistance(GameEngine):
    """
    Game Simulator for keep distance mode. 保持距离模式的游戏模拟器

    The goal of the rabbit is to keep certain distance with the gun  兔子的目标是和枪保持特定的距离
    """
    def __init__(self, training, screen_size, rabbit_size, gun_size, rabbit_speed, gun_speed,
                 history_size, max_time_step, debug, desired_goal, lower_goal=None, higher_goal=None):   # 这里所谓的期望目标、和最低最高目标都指的是距离
        """
        Initialization, see `GameEngine.__init__()` for detail
        """
        if lower_goal is not None and higher_goal is not None:
            if desired_goal < lower_goal or desired_goal > higher_goal:
                raise ValueError('desired goal {} should be in the range [{}, {}]',   # 期望目标应该在最低目标和最高目标之间
                                 desired_goal, lower_goal, higher_goal)

        self.lower_distance = max(0., desired_goal - 0.1)
        if lower_goal is not None:
            self.lower_distance = lower_goal

        self.higher_distance = min(np.sqrt(2), desired_goal + 0.1)√√
        if higher_goal is not None:
            self.higher_distance = higher_goal              # 让距离保持在最高和最低距离之间，而且最多不超过， [0, √2]，这个范围不太大

        super(GameKeepDistance, self).__init__(training, screen_size, rabbit_size, gun_size,
                                               rabbit_speed, gun_speed, history_size,
                                               max_time_step, debug)

    def reset(self):
        """
        Reset ebvironment. See `GameState.reset()` for detail.    重置环境， GameState.reset()这个函数没看见过， 不知道放在哪里
        """
        state = GameEngine.reset(self)
        self.distance = float(np.linalg.norm(self.rabbit_loc - self.gun_loc)) / self.screen_size
        return state    # reset返回的就是状态

    def get_reward_terminal(self):
        """
        Overwrite base class method.  重写基类方法， 返回元组
        :return: tuple of (reward, terminal, info)
        """
        self.distance = float(np.linalg.norm(self.rabbit_loc - self.gun_loc)) / self.screen_size   #  典型的距离计算方式
        reward = 1 if self.lower_distance <= self.distance <= self.higher_distance else -1    
        # 如果距离在合适范围内， 就奖励， 所以就是，为了限制每种策略，训练每种策略， 那我们就利用奖惩的方式。符合要求，我们就奖励。 当然，兔子也不能撞墙，所以也是奖惩的一部分。
        # terminal = 1 if self.frame_cnt >= self.max_time_step or self.rabbit_hit_boundary else 0
        if self.rabbit_hit_boundary:   # 兔子撞墙就惩罚
            reward += -1
        terminal = 1 if self.frame_cnt >= self.max_time_step else 0

        info = {'distance': self.distance}   # 这里的信息，就是distance，也就是从信息就能看出我们是采用了哪种策略。
 
        return reward, terminal, info

    def add_text_render(self, location=(0, 0)):
        """
        overwrite `GameEngine.add_text_render()`
        """
        self.text = 'distance: {:.2f}, goal: [{:.2f}, {:.2f}]'.format(self.distance,
                                                                      self.lower_distance, self.higher_distance)
        super(GameKeepDistance, self).add_text_render()    #  渲染文字信息， 显示在屏幕上， 就能知道我们采用何种模式
