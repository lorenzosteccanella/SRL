import gym
import numpy as np


class RingEnv(gym.Env):

    def __init__(self, max_n_steps_episode=10):
        """A simple ring environment"""
        self.state_space = gym.spaces.Box(0, 2, shape=(1,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)
        self.position = 0
        self.max_n_steps_episode = max_n_steps_episode
        self.steps = 0
        
    def one_hot_encode(self, x):
        return np.eye(3)[x]

    def reset(self):
        self.position = 0
        self.steps = 0
        return self.one_hot_encode(self.position)

    def step(self, a):
        if a == 0:
            self.position = self.position
        elif self.position == 0 and a == 1:
            self.position = 1
        elif self.position == 1 and a == 1:
            self.position = 2
        elif self.position == 2 and a == 1:
            self.position = 0
        elif self.position == 0 and a == 2:
            self.position = 2
        elif self.position == 2 and a == 2:
            self.position = 1
        elif self.position == 1 and a == 2:
            self.position = 1

        self.steps += 1
        done = False
        if self.steps == self.max_n_steps_episode:
            done = True

        return (self.one_hot_encode(self.position), # State
                0,
                done, # Done flag
                dict()) # Additional info





