import gym
from torch.utils.data import DataLoader
from Utils.Utils import *
from Problem_formulation.Models import LearnedModel
from Problem_formulation.train_action import minibatch_deep_A_ER, minibatch_deep_A
from Problem_formulation.train_repr import minibatch_deep_WE
import torch

config = {
    "seed":1,

    # dataset and environment parameters
    "load_trajectories": True,
    "n_of_trajectories": 100,
    "shuffle": True,
    "collate_fn": my_collate_v2,
    "obs_noise": False,
    "len_window_o": 200,
    "c_window_len": 0,
    "d_sampler": 1,
    "percent_random_sample": 1,
    "ub_w_dist": 1,
    "path_dataset": "/home/lorenzo/Documenti/UPF/MDP_temporal_representation/Data/dataset_MountainCar.pt",
    "path_trajectory": "/home/lorenzo/Documenti/UPF/MDP_temporal_representation/Data/trajectory_MountainCar.npy",

    # wandb params
    "wandb_record": False,
    "wandb_group": "MountainCar",

    # training params
    "batch_size": 512,
    "batch_size_action_mod": 128,
    "out_d": 64,
    "in_d": 2,
    "in_a_d": 3,
    "epochs": 300,
    "max_n_steps": 40000,
    "epochs_action_mod": 300,
    "max_n_steps_action_mod": 40000,
    "path_model": "/home/lorenzo/Documenti/UPF/MDP_temporal_representation/Models/MountainCar_action_states_dictionary",
    "constr_weight": 1,
    "l_rate": 0.001,
    "weight_interpol": 0.4,
    "amsgrad": True
}

def training_model(config):
    # Fixing all seed for reproducibility
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])

    if config["load_trajectories"]:
        trajectories = load_action_trajectories(config["path_trajectory"])
    else:
        env = gym.make("MountainCar-v0")
        env.seed(config["seed"])
        trajectories, states = dqn_mountain_car_collect_trajectories(env,
                                                                  n_of_trajectories=config["n_of_trajectories"],
                                                                  obs_noise=config["obs_noise"],
                                                                  action_state=True,
                                                                  discrete=False,
                                                                  scale_factor=1.)
        save_action_trajectories(trajectories, config["path_trajectory"])

    dataset = dataset_w_c(trajectories, len_window_o=config["len_window_o"], c_window_len=config["c_window_len"],
                          d_sampler=config["d_sampler"], percent_random_sample=config["percent_random_sample"],
                          ub_w_dist=config["ub_w_dist"])

    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=config["shuffle"], collate_fn=config["collate_fn"])
    encoder = LearnedModel(config["in_d"], config["out_d"], config["in_a_d"])

    if config["wandb_record"]:
        run = wandb.init(project='MDP_DIST', entity='lsteccanella', config=config,
                         group=config["wandb_group"], settings=wandb.Settings(start_method="fork"))
        wandb.run.name = "state_action_encoding"
        wandb.watch(encoder, criterion=None, log="all", log_freq=100, idx=None, log_graph=(False))

    loss1 = minibatch_deep_WE(dataloader, encoder, config)
    loss2 = minibatch_deep_A(dataloader, encoder, config)
    torch.save(encoder.state_dict(), config["path_model"])

    print("obj_result:", loss1, "obj2_result:", loss2)
    encoder.eval()
    z_states = []
    z_states2 = []
    states = []
    annotations = []
    for j, trajectory in enumerate(trajectories):
        for state in trajectory:
            s = torch.FloatTensor([state[0]])
            a = torch.FloatTensor([state[1]])
            z_states.append(encoder.encode_next_state(s, a).detach().numpy()[0])
            z_states2.append(encoder.encode_state(s).detach().numpy()[0])

    plot_result3(z_states, np.zeros(len(z_states)), config["out_d"], 0, False)

training_model(config)