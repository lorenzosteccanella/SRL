import os
import sys
dirpath = os.path.dirname(os.path.abspath(__file__))
dirname = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(dirpath))

from Repr_learning.KeyDoorPos_PER import training_model
from Utils.Utils import *


seed = int(sys.argv[1])

config = {
    # dataset and environment parameters
    "seed": seed,
    "load_trajectories": False,
    "n_of_trajectories": 10,
    "shuffle": True,
    "collate_fn": my_collate_v2,
    "obs_noise": False,
    "len_window_o": 40,
    "c_window_len": 0,
    "d_sampler": 1,
    "percent_random_sample": 1,
    "ub_w_dist": 1,
    "path_dataset": dirname + "/Data/dataset_KeyDoorPos.pt",
    "path_trajectory": dirname + "/Data/trajectory_KeyDoorPos.npy",

    # wandb params
    "wandb_record": False,
    "wandb_group": "KeydoorGridWorld",

    # training params
    "batch_size": 512,
    "batch_size_action_mod": 128,
    "out_d": 2,
    "in_d": 4,
    "in_a_d": 5,
    "epochs": 300,
    "max_n_steps": 10000,
    "epochs_action_mod": 300,
    "max_n_steps_action_mod": 1000,
    "path_model": dirname + "/Models/Minigrid_action_states_dictionary",
    "constr_weight": 1,
    "l_rate": 0.001,
    "amsgrad": True

}

training_model(config)