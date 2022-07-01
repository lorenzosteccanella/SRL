from copy import deepcopy

import gym
import numpy as np

env = gym.make("Reacher-v2")

env.seed(0)

action_fn = lambda: (np.random.rand(1, 2) * 2) - 1
env.reset()
s = deepcopy(env.get_body_com("fingertip"))
goal = env.get_body_com("target")
done = False
n_steps = 0
while not done:
    s_, reward, done, info = env.step(action_fn())
    s_ = deepcopy(env.get_body_com("fingertip"))
    n_steps += 1
    print(s, s_, goal)
    s = s_