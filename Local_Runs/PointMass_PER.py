import os
import sys

from Envs.PointmassEnv import PointmassEnv

dirpath = os.path.dirname(os.path.abspath(__file__))
dirname = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(dirpath))

from Repr_learning.PointMass_PER import training_model
from Utils.Utils import *
from Planning.Planning_alg import *
from Problem_formulation.Models import LearnedModel


seed = 1#int(sys.argv[1])

config = {
    # dataset and environment parameters
    "seed": seed,
    "load_trajectories": False,
    "n_of_trajectories": 100,
    "max_n_steps_env": 50,
    "shuffle": True,
    "collate_fn": my_collate_v2,
    "obs_noise": False,
    "len_window_o": 40,
    "c_window_len": 0,
    "d_sampler": 1,
    "percent_random_sample": 1,
    "ub_w_dist": 1,
    "path_dataset": dirname + "/Data/dataset_PointMass.pt",
    "path_trajectory": dirname + "/Data/trajectory_PointMass.npy",

    # wandb params
    "wandb_record": False,
    "wandb_group": "KeydoorGridWorld",

    # training params
    "batch_size": 512,
    "batch_size_action_mod": 64,
    "out_d": 64,
    "in_d": 2,
    "in_a_d": 5,
    "epochs": 300,
    "max_n_steps": 100000,
    "epochs_action_mod": 300,
    "max_n_steps_action_mod": 100000,
    "path_model": dirname + "/Models/PointMass_action_states_dictionary",
    "constr_weight": 1,
    "l_rate": 0.001,
    "amsgrad": True

}

encoder = training_model(config)
encoder.eval()
env = PointmassEnv()
env.seed(config["seed"])
planner = Planner(encoder, list(range(env.action_space.n)), 20, 5)

n_epochs_evaluation = 200

distances = []
len_path = []
for i in range(n_epochs_evaluation):
    s = env.reset()
    desired_goal = env.get_goal()
    steps = 0
    states = []
    len = 0
    while steps < config["max_n_steps_env"]:
        states.append(s)
        len += np.linalg.norm(s - desired_goal)
        shortest_path_action = planner.multi_step_lookahead(s, [desired_goal])
        s_, r, done, info = env.step(shortest_path_action); steps += 1
        print(s, desired_goal)
        s = s_
    len_path.append(len)
    distances.append(np.linalg.norm(states[-1] - desired_goal))

print('Mean Distance to Goal:  %.3f' % np.mean(distances))
print('Min Distance to Goal:  %.3f' % np.min(distances))
print('Max Distance to Goal:  %.3f' % np.max(distances))
print('Mean Distance Traversed to Goal: %.3f' % np.mean(len_path))
