import gym
import numpy as np
from multiworld import *

register_all_envs()

env = gym.make("SawyerReachXYZEnv-v1")
env.seed(0)
np.random.seed(0)
for i in range(50):
    state_info = env.reset()
    s = state_info["observation"]
    goal = state_info["desired_goal"]
    max_steps = 200
    steps = 0
    goal_reached_fn = lambda s, goal: np.linalg.norm(s-goal, 2) < 0.01
    while steps < max_steps or goal_reached_fn(s, goal):
        #a = (np.random.rand(1, 3) * 2) - 1
        a = np.array([1, 0, 0])
        state_info, reward, done, info = env.step(a); steps += 1
        s_ = state_info["observation"]
        print(s_, goal, np.linalg.norm(s_-goal, 2), reward)
        s = s_
        #env.render()

    print(s, goal)