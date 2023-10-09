import numpy as np
import torch
from Planning.Planning_alg import Planner
import gym
import os
dirname = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

env = gym.make("CartPole-v0")

model = torch.load(dirname + "/Saved_models_v2/CartPole_full_model")
model.eval()

seed = 0
env.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

planner = Planner(model, list(range(env.action_space.n)), max_n_rand_traj=40, max_horizon=4)

steps = 0
sum_r = 0
desired_goal = [[0, 0, 0, 0]]
z_goal = planner.get_z_goal_states(desired_goal)
print(z_goal)
while steps < 100000:
    s = env.reset()
    while True:
        p_a = planner.prob_multi_step_lookahead(s, z_goal, 0.01, verbose=False)
        s_, r, done, info = env.step(p_a); steps += 1

        env.render()

        sum_r += r
        s = s_
        if done: break

    print(steps, sum_r, s, desired_goal)
    sum_r = 0