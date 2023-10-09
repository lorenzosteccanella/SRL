import sys
import os

dirpath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(dirpath)

import numpy as np
import torch
from Envs.PointmassEnv import PointmassEnv
from Planning.Planning_alg import Planner
import wandb

seed = int(sys.argv[1])

run = wandb.init(project='MAD_SemiNorm', entity='lsteccanella',
                 group="Planning_PointMass_L1", settings=wandb.Settings(start_method="fork"))
wandb.run.name = "planning_pointmass_l1_" + str(seed)

env = PointmassEnv(max_n_steps_episode=50)
action_space = list(range(env.action_space.n))

env.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

model = torch.load(dirpath + "/Saved_models/PointMass_L1_" + str(seed))
model.eval()

planner = Planner(model, list(range(env.action_space.n)), max_n_rand_traj=10, max_horizon=2)

sum_r = 0
steps = 0
for i in range(1000):
    s = env.reset()
    desired_goal = env.get_goal()
    z_goal = planner.get_z_goal_states(np.array([desired_goal]))
    while True:
        p_a = planner.prob_multi_step_lookahead(s, z_goal, 0.01, verbose=False)
        s_, r, done, info = env.step(p_a); steps += 1

        sum_r += r
        s = s_
        if done: break

    wandb.log({"Reward": sum_r, "Steps": steps})
    sum_r = 0
    steps = 0




