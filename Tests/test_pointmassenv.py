from Envs.PointmassEnv import PointmassEnv
import numpy as np

env = PointmassEnv()
s = env.reset()

for step in range(200):
    a = np.random.choice(env.action_space.n)
    s_, reward, done, info = env.step(a)