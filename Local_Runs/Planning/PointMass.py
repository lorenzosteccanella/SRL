import numpy as np
import torch
from Envs.PointmassEnv import PointmassEnv
import os
from Planning.Planning_alg import Planner

dirname = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

env = PointmassEnv(max_n_steps_episode=50)
action_space = list(range(env.action_space.n))

model = torch.load(dirname + "/Saved_models_v2/PointMass_full_model")
model.eval()

seed = 0
env.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

planner = Planner(model, list(range(env.action_space.n)), max_n_rand_traj=10, max_horizon=1)

steps = 0
sum_r = 0
while steps < 100000:
    s = env.reset()
    desired_goal = env.get_goal()
    z_goal = planner.get_z_goal_states(np.array([desired_goal]))
    while True:
        p_a = planner.prob_multi_step_lookahead(s, z_goal, 0.1, verbose=False)
        s_, r, done, info = env.step(p_a); steps += 1

        sum_r += r
        s = s_
        if done: break

    print(steps, sum_r, s, desired_goal)
    sum_r = 0



