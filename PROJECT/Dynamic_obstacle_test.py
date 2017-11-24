from __future__ import division
import numpy as np
from collections import deque

import pygame


RED    = (255, 0,   0  )
GREEN  = (0,   255, 0  )
BLUE   = (0,   0,   255)
BLACK  = (0,   0,   0  )
WHITE  = (255, 255, 255)
YELLOW = (255, 255, 0)

# predefined colors
PREDEFINED_COLORS = {'red'   : RED,
                     'green' : GREEN,
                     'blue'  : BLUE,
                     'black' : BLACK,
                     'white' : WHITE,
                     'yellow': YELLOW}


class Car(object):
    def __init__(self, size, color, speed):
        self.size = size
        self.color = PREDEFINED_COLORS[color]
        self.speed = speed
        self.set_pos(np.zeros((2,)))

    def set_pos(self, new_pos):
        self.pos = new_pos.copy()
        self.prev_pos = self.pos.copy()

    def reset_pos(self):
        """ reset postion. use when the car hit the obstacle """
        self.pos = self.prev_pos.copy()

    def step(self, action):
        assert action < 4 and action >= 0, 'invalid action: {}'.format(action)
        # move up
        self.prev_pos = self.pos.copy()
        if action == 0:
            self.pos[1] -= self.speed
        # move down
        elif action == 1:
            self.pos[1] += self.speed
        # move left
        elif action == 2:
            self.pos[0] -= self.speed
        else:
            self.pos[0] += self.speed

    def render(self, surface):
        width, height = self.size, self.size
        left, top = int(self.pos[0]), self.pos[1]
        pygame.draw.rect(surface, self.color, (left, top, width, height), 0)

class Obstacle(object):
    """ Obstacle """
    def __init__(self, size, color, pos):
        self.size = size
        self.color = PREDEFINED_COLORS[color]
        self.pos = pos

    def render(self, surface):
        width, height = self.size, self.size
        left, top = self.pos[0], self.pos[1]
        pygame.draw.rect(surface, self.color, (left, top, width, height), 0)

class Maze(object):
    def __init__(self, world_size, grid_size, history_size):
        self.world_size = world_size
        self.grid_size = grid_size
        self.car = Car(grid_size, 'red', self.grid_size)
        self.n = self.world_size // self.grid_size
        self.history = deque(maxlen=history_size)
        self.start = np.zeros((2,))
        self.end = np.ones((2,)) * (self.n-1)
        
        self.screen = None

    def _make_obstacles(self):
        self.obstacles = []
        # let's make 3 obstacles
        # the grid cant be at the start/end
        for i in range(3):
            while True:
                pos = np.random.randint(0, self.n, size=2)
                if not ((pos == self.start).all() or (pos == self.end).all()):
                    break
            pos *= self.grid_size
            self.obstacles.append(Obstacle(self.grid_size, 'blue', pos))

    def _make_state(self):
        state = np.concatenate(list(self.history))
        state /= self.grid_size
        return state

    def reset(self):
        self.car.set_pos(self.start)
        self._make_obstacles()
        self.history.clear()

        for i in range(self.history.maxlen):
            self.history.append(self.car.pos.copy())

        return self._make_state()

    def _check_hit_boundary(self):
        cx, cy = self.car.pos
        cx = cx // self.grid_size
        cy = cy // self.grid_size
       
        if cx < 0 or cx >= self.n or cy < 0 or cy >= self.n:
            return True
        return False

    def _check_hit_obstacle(self):
        cx, cy = self.car.pos
        cx = cx // self.grid_size
        cy = cy // self.grid_size
        
        for obs in self.obstacles:
            ox, oy = obs.pos
            ox = ox // self.grid_size
            oy = oy // self.grid_size
            if ox == cx and oy == cy:
                return True
        return False

    def _reach_end(self):
        cx, cy = self.car.pos
        cx = cx // self.grid_size
        cy = cy // self.grid_size
        if cx == self.end[0] and cy == self.end[1]:
            return True
        return False

    def step(self, action):
        self.action = action
        self.car.step(action)
        
        self.car.hit_boundary = self._check_hit_boundary()
        self.car.hit_obstacle = self._check_hit_obstacle()

        if self.car.hit_boundary or self.car.hit_obstacle:
            self.car.reset_pos()
        
        if self._reach_end():
            terminal = True
            reward = +1
        else:
            terminal = False
            reward = 0

        self.history.append(self.car.pos.copy())
        state_tp1 = self._make_state()

        # info is something for debugging
        info = {}

        return state_tp1, reward, terminal, info

    def _init_screen(self):
        """ initialization of screen """
        self.screen = pygame.display.set_mode((self.world_size, self.world_size))
        pygame.display.set_caption('maze world')
    
    def render(self):
        """ Render """
        if self.screen is None:
            self._init_screen()
            self.bg_color = BLACK
            self.FPS = 30
            self.FPSCLOCK = pygame.time.Clock()
            self.Font = pygame.font.SysFont('Comic Sans MS', 30)
        pygame.event.pump() 

        pygame.draw.rect(self.screen, PREDEFINED_COLORS['red'], (0, 0, 60, 60), 0)
        # fill the bg
        self.screen.fill(self.bg_color)
        self.car.render(self.screen)
        for obs in self.obstacles:
            obs.render(self.screen)

        pygame.display.update()
        self.FPSCLOCK.tick(self.FPS)

if __name__ == '__main__':
    pygame.init()
    env = Maze(300, 60, 5)
    while True:
        state = env.reset()
        print (state)
        env.render()
        terminal = False
        while not terminal:
            action = np.random.randint(4)
            state, reward, terminal, _ = env.step(action)
            print (state)
            env.render()
