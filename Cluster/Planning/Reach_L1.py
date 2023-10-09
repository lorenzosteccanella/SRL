import sys
import os

dirpath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(dirpath)

import numpy as np
import torch
from Planning.Planning_alg import Planner
import gymnasium as gym
import wandb

seed = int(sys.argv[1])

run = wandb.init(project='MAD_SemiNorm', entity='lsteccanella',
                 group="Planning_Reach_L1", settings=wandb.Settings(start_method="fork"))
wandb.run.name = "planning_reach_L1_" + str(seed)

env = gym.make('FetchReach-v2', max_episode_steps=100)
action_space = list(np.array(
  np.meshgrid(np.arange(-1, 1.2, 0.2), np.arange(-1, 1.2, 0.2), np.arange(-1, 1.2, 0.2), 0)).T.reshape(-1, 4))

np.random.seed(seed)
torch.manual_seed(seed)

model = torch.load(dirpath + "/Saved_models/Reach_L1_" + str(seed))
model.eval()

planner = Planner(model, action_space, max_n_rand_traj=20, max_horizon=0)

sum_r = 0
steps = 0
for i in range(1000):
    s, _ = env.reset(seed=seed)
    desired_goal = s["desired_goal"]
    s = s["observation"][:3]
    z_goal = planner.get_z_goal_states(np.array([desired_goal]))
    while True:
        p_a = planner.cont_prob_multi_step_lookahead(s, z_goal, 0.01, verbose=False)
        s_, r, _, done, info = env.step(p_a); steps += 1
        s_ = s_["observation"][:3]
        sum_r += r
        s = s_

        if done or r==0: break

    wandb.log({"Reward": sum_r, "Steps": steps})
    sum_r = 0
    steps = 0