import os
import sys
dirpath = os.path.dirname(os.path.abspath(__file__))
dirname = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(dirpath))

from Repr_learning.Acrobot_PER import training_model
from Utils.Utils import *


seed = int(sys.argv[1])

config = {
    "seed": seed,

    # dataset and environment parameters
    "load_trajectories": True,
    "n_of_trajectories": 100,
    "shuffle": True,
    "collate_fn": my_collate_v2,
    "obs_noise": False,
    "len_window_o": None,
    "c_window_len": 0,
    "d_sampler": 1,
    "percent_random_sample": 1,
    "ub_w_dist": 1,
    "path_dataset": dirname + "/Data/dataset_Acrobot.pt",
    "path_trajectory": dirname + "/Data/trajectory_Acrobot.npy",

    # wandb params
    "wandb_record": False,
    "wandb_group": "Acrobot",

    # training params
    "batch_size": 512,
    "batch_size_action_mod": 128,
    "out_d": 64,
    "in_d": 6,
    "in_a_d": 3,
    "epochs": 300,
    "max_n_steps": 40000,
    "epochs_action_mod": 300,
    "max_n_steps_action_mod": 40000,
    "path_model": dirname + "/Models/Acrobot_action_states_dictionary",
    "constr_weight": 1,
    "l_rate": 0.001,
    "amsgrad": True

}

training_model(config)