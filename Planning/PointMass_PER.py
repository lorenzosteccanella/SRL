from Planning.Planning_alg import *
from Problem_formulation.Models import LearnedModel
from Envs.PointmassEnv import PointmassEnv
import numpy as np
import wandb

env = PointmassEnv(max_n_steps_episode=50)
action_space = list(range(env.action_space.n))
encoder = LearnedModel(2, 64, len(action_space))
encoder.load_state_dict(
    torch.load("/Saved_models/PointMass_action_states_dictionary_0"))

seed = 0
env.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

planner = Planner(encoder, list(range(env.action_space.n)), 10, 5)

wandb_group = "PointMass_Planning"
wandb_name = "PointMass_Planning_seed_" + str(seed)

run = wandb.init(project='MDP_DIST', entity='lsteccanella',
                 group=wandb_group, settings=wandb.Settings(start_method="fork"))
wandb.run.name = wandb_name

steps = 0
while steps < 100000:
    s = env.reset()
    desired_goal = env.get_goal()
    z_goals = planner.get_z_goal_states([desired_goal])
    len = 0
    while True:
        p_a = planner.prob_multi_step_lookahead(s, z_goals, 0.8, verbose=False)
        s_, r, done, info = env.step(p_a); steps += 1
        s = s_
        if done: break

    wandb.log({"dist_goal": np.linalg.norm(s - desired_goal, 1)},
              step=steps)