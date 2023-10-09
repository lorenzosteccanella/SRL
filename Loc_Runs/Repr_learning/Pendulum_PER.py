import os
import sys
dirpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(dirpath))

from Repr_learning.Pendulum_PER import training_model
from Utils.Utils import *


seed = int(sys.argv[1])

config = {
    "seed": seed,

    # dataset and environment parameters
    "load_trajectories": False,
    "n_of_trajectories": 100,
    "shuffle": True,
    "collate_fn": my_collate_v2,
    "obs_noise": False,
    "len_window_o": None,
    "c_window_len": 0,
    "d_sampler": 1,
    "percent_random_sample": 1,
    "ub_w_dist": 1,
    "path_dataset": dirname + "/Data/dataset_Pendulum.pt",
    "path_trajectory": dirname + "/Data/trajectory_Pendulum.npy",

    # wandb params
    "wandb_record": True,
    "wandb_group": "Pendulum_3",

    # training params
    "batch_size": 512,
    "batch_size_action_mod": 128,
    "out_d": 64,
    "in_d": 3,
    "in_a_d": 1,
    "epochs": 300,
    "max_n_steps": 100000,
    "epochs_action_mod": 300,
    "max_n_steps_action_mod": 10000,
    "path_model": dirname + "/Models/Pendulum_action_states_dictionary_3",
    "constr_weight": 1,
    "l_rate": 0.0005,
    "amsgrad": True

}

training_model(config)
