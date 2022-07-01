import gym
from torch.utils.data import DataLoader
from Utils.Utils import *
from Problem_formulation.Models import LearnedModel
from Problem_formulation.train_action import minibatch_deep_A_ER, minibatch_deep_A
from Problem_formulation.train_repr import minibatch_deep_PER
import torch
import os
dirname = os.path.dirname(os.path.dirname(__file__))

config = {
    "seed": 2,

    # dataset and environment parameters
    "load_trajectories": False,
    "n_of_trajectories": 200,
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
    "wandb_record": False,
    "wandb_group": "Pendulum",

    # training params
    "batch_size": 512,
    "batch_size_action_mod": 128,
    "out_d": 64,
    "in_d": 3,
    "in_a_d": 1,
    "epochs": 300,
    "max_n_steps": 40000,
    "epochs_action_mod": 300,
    "max_n_steps_action_mod": 10000,
    "path_model": dirname + "/Models/Pendulum_action_states_dictionary",
    "constr_weight": 1,
    "l_rate": 0.0005,
    "amsgrad": True

}

def training_model(config):
    # Fixing all seed for reproducibility
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])

    if config["load_trajectories"]:
        trajectories = load_action_trajectories(config["path_trajectory"])
        print(len(trajectories))

    else:
        env = gym.make("Pendulum-v0")
        env.seed(config["seed"])
        trajectories, states = ddpg_pendulum_collect_trajectories(env,
                                                                  n_of_trajectories=config["n_of_trajectories"],
                                                                  obs_noise=config["obs_noise"],
                                                                  action_state=True,
                                                                  discrete=False,
                                                                  scale_factor=1.)
        save_action_trajectories(trajectories, config["path_trajectory"])

    per = PER(trajectories, replay_size=1e7, len_window_o=config["len_window_o"], c_window_len=config["c_window_len"],
                          d_sampler=config["d_sampler"], percent_random_sample=config["percent_random_sample"],
                          ub_w_dist=config["ub_w_dist"])

    encoder = LearnedModel(config["in_d"], config["out_d"], config["in_a_d"])

    if config["wandb_record"]:
        run = wandb.init(project='MDP_DIST', entity='lsteccanella', config=config,
                         group=config["wandb_group"], settings=wandb.Settings(start_method="fork"))
        wandb.run.name = "repr_learning_seed_" + str(config["seed"])
        wandb.watch(encoder, criterion=None, log="all", log_freq=100, idx=None, log_graph=(False))

    loss1 = minibatch_deep_PER(per, encoder, config)

    dataset = dataset_w_c(trajectories, len_window_o=config["len_window_o"], c_window_len=config["c_window_len"],
                          d_sampler=config["d_sampler"], percent_random_sample=config["percent_random_sample"],
                          ub_w_dist=config["ub_w_dist"])

    dataloader = DataLoader(dataset, batch_size=config["batch_size_action_mod"], shuffle=config["shuffle"], collate_fn=config["collate_fn"])

    loss2 = minibatch_deep_A(dataloader, encoder, config)
    torch.save(encoder.state_dict(), config["path_model"]+"_"+str(config["seed"]))

    print("obj_result:", loss1, "obj2_result:", loss2)

    return encoder