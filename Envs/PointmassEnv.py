import gym
import numpy as np


class PointmassEnv(gym.Env):
    def __init__(self, max_n_steps_episode):
        # 2-d coordinates
        self.state_space = gym.spaces.Box(-10, 10, shape=(2,), dtype=np.float32)
        self.position = np.zeros(2)
        self.s_goal = np.array([0, 0])
        # Up, Right, Down, Left, Stay
        self.action_space = gym.spaces.Discrete(5)
        self.max_n_steps_episode = max_n_steps_episode
        self.steps = 0
        self.wh = 10

    def reset(self):
        self.sample_goal()
        self.position = np.zeros(2)
        self.steps = 0
        return self.position

    def action_to_direction(self, a):
        actions = [
            np.array([0, 1]),  # Up
            np.array([1, 0]),  # Right
            np.array([0, -1]),  # Down
            np.array([-1, 0]),  # Left,
            np.array([0, 0]),  # Stay still
        ]
        return actions[a]

    def step(self, a):
        direction = self.action_to_direction(a)
        step = 0.5 * direction #+ np.random.randn()# Take a noisy step in direction
        self.position = self.position + step; self.steps += 1
        self.position = np.clip(self.position, -self.wh, self.wh)  # Clip to prevent object from escaping
        done = False
        if np.array_equal(self.position, self.s_goal):
            done = True

        if self.steps == self.max_n_steps_episode:
            done = True

        return (self.position,  # State
                0,  # Reward (not necessary for GCSL)
                done,  # Done flag
                dict())  # Additional info

    def reset_position(self, pos):
        self.position = pos

    def sample_goal(self):
        self.s_goal = np.array([0, 0])
        while np.array_equal(self.s_goal, np.array([0, 0])):
            self.s_goal = np.round(np.random.rand(2) * (self.wh*2) - self.wh, 0)  # Sample uniformly from [-10, 10]

    def get_goal(self):
        return self.s_goal