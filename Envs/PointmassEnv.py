from copy import deepcopy

import gym
import numpy as np


class PointmassEnv(gym.Env):
    def __init__(self, max_n_steps_episode):
        # 2-d coordinates
        self.state_space = gym.spaces.Box(0, 10, shape=(2,), dtype=np.float32)
        self.position = np.zeros(2)
        self.s_goal = np.array([0, 0])
        # Up, Right, Down, Left, Stay
        self.action_space = gym.spaces.Discrete(5)
        self.max_n_steps_episode = max_n_steps_episode
        self.steps = 0

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

    def forward(self, s, a):
        direction = self.action_to_direction(a)
        step = 1 * direction
        s_ = s + step
        self.steps += 1
        s_ = np.clip(s_, self.state_space.low[0], self.state_space.high[0])
        return s_

    def step(self, a):
        self.position = self.forward(self.position, a)
        # Clip to prevent object from escaping
        done = False
        if np.array_equal(self.position, self.s_goal):
            done = True

        if self.steps == self.max_n_steps_episode:
            done = True

        return (self.position,  # State
                -1,  # Reward (not necessary for GCSL)
                done,  # Done flag
                dict())  # Additional info

    def compute_MAD_distance(self):
        """
        Uses the Floyd-Warshall algorithm to compute the distance between all pairs of states in the state space.
        Returns:
            mad_distance (np.ndarray): Matrix of distances between all pairs of states.
        """

        # Initialize the distance matrix
        width, height = int(self.state_space.high[0]), int(self.state_space.high[1])
        all_states = [(i, j) for i in range(width+1) for j in range(height+1)]
        base_dict = dict.fromkeys(all_states, np.inf)
        mad_distance = {}
        for i in all_states:
            mad_distance[i] = deepcopy(base_dict)

        # adjacent vertex we set value to 1
        for i in all_states:
            for j in all_states:
                # set the value of adjacent vertex to 1:
                if not i == j:
                    # test if the state is reachable in one step
                    a = 0
                    while a in range(self.action_space.n):
                        if tuple(self.forward(np.array(i), a)) == j:
                            mad_distance[i][j] = 1
                            break
                        a += 1
                else:
                    mad_distance[i][j] = 0

        # compute shortest paths
        for i in all_states:
            for j in all_states:
                for k in all_states:
                    if mad_distance[i][j] > mad_distance[i][k] + mad_distance[k][j]:
                        mad_distance[i][j] = mad_distance[i][k] + mad_distance[k][j]

        return mad_distance

    def reset_position(self, pos):
        self.position = pos

    def sample_goal(self):
        self.s_goal = np.array([0, 0])
        while np.array_equal(self.s_goal, np.array([0, 0])):
            self.s_goal = np.round(np.random.rand(2) * (self.state_space.high[0]) - self.state_space.low[0], 0)  # Sample uniformly from [low, high]

    def get_goal(self):
        return self.s_goal


class AsyPointmassEnv(gym.Env):
    def __init__(self, max_n_steps_episode):
        # 2-d coordinates
        self.state_space = gym.spaces.Box(0, 10, shape=(2,), dtype=np.float32)
        self.position = np.zeros(2)
        self.s_goal = np.array([0, 0])
        # Up, Right, Down, Left, Stay
        self.action_space = gym.spaces.Discrete(5)
        self.max_n_steps_episode = max_n_steps_episode
        self.steps = 0

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

    def forward(self, s, a):
        direction = self.action_to_direction(a)
        if a == 0 or a == 1:
            step = 1 * direction
        elif a == 2 or a == 3:
            step = 2 * direction
        elif a == 4:
            step = 0 * direction
        s_ = s + step
        self.steps += 1
        s_ = np.clip(s_, self.state_space.low[0], self.state_space.high[0])
        return s_

    def step(self, a):
        self.position = self.forward(self.position, a)
        done = False
        if np.array_equal(self.position, self.s_goal):
            done = True

        if self.steps == self.max_n_steps_episode:
            done = True

        return (self.position,  # State
                -1,  # Reward (not necessary for GCSL)
                done,  # Done flag
                dict())  # Additional info

    def compute_MAD_distance(self):
        """
        Uses the Floyd-Warshall algorithm to compute the distance between all pairs of states in the state space.
        Returns:
            mad_distance (np.ndarray): Matrix of distances between all pairs of states.
        """

        # Initialize the distance matrix
        width, height = int(self.state_space.high[0]), int(self.state_space.high[1])
        all_states = [(i, j) for i in range(width+1) for j in range(height+1)]
        base_dict = dict.fromkeys(all_states, np.inf)
        mad_distance = {}
        for i in all_states:
            mad_distance[i] = deepcopy(base_dict)

        # adjacent vertex we set value to 1
        for i in all_states:
            for j in all_states:
                # set the value of adjacent vertex to 1:
                if not i == j:
                    # test if the state is reachable in one step
                    a = 0
                    while a in range(self.action_space.n):
                        if tuple(self.forward(np.array(i), a)) == j:
                            mad_distance[i][j] = 1
                            break
                        a += 1
                else:
                    mad_distance[i][j] = 0

        # compute shortest paths
        for i in all_states:
            for j in all_states:
                for k in all_states:
                    if mad_distance[i][j] > mad_distance[i][k] + mad_distance[k][j]:
                        mad_distance[i][j] = mad_distance[i][k] + mad_distance[k][j]


        return mad_distance

    def reset_position(self, pos):
        self.position = pos

    def sample_goal(self):
        self.s_goal = np.array([0, 0])
        while np.array_equal(self.s_goal, np.array([0, 0])):
            self.s_goal = np.round(np.random.rand(2) * (self.state_space.high[0]) - self.state_space.low[0], 0)  # Sample uniformly from [low, high]

    def get_goal(self):
        return self.s_goal