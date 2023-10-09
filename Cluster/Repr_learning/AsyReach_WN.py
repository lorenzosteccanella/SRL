import sys
import os

dirpath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(dirpath)

from tqdm import trange
from Utils.Utils import *
from Models.Models import *
import gymnasium as gym

seed = int(sys.argv[1])

config= {
    "seed": seed,
    "path_trajectory": dirpath + "/Data/trajectory_AsyReach.npy",
    "in_d": 3,
    "out_d": 3,
    "in_dist_d": 8,
    "out_dist_d": 8,
    "dist_type": "WideNorm",
    "l_rate": 0.0005,
    "amsgrad": True,
    "batch_size_o": 64,
    "batch_size_c": 512,
    "max_dist_obj": None,
    "dist_discount": 1,
    "max_dist_con": 1,
    "weight_constrains": 10,
    "gradient_steps": 10000,
    "in_a_d": 4,
    "l_rate_action": 0.0005,
    "amsgrad_action": True,
    "batch_size_action": 128,
    "gradient_steps_action": 10000,

    "wandb_group": "AsyReach_WN",

    "model_path": dirpath + "/Saved_models/AsyReach_WN_" + str(seed)
}

np.random.seed(config["seed"])
torch.manual_seed(config["seed"])
random.seed(config["seed"])

run = wandb.init(project='MAD_SemiNorm', entity='lsteccanella', config=config,
                 group=config["wandb_group"], settings=wandb.Settings(start_method="fork"))
wandb.run.name = "repr_learning_seed_" + str(config["seed"])

# env = gym.make('FetchReach-v2', max_episode_steps=100)

# trajectories = collect_continuous_action_trajectories(env, 100, None)
# save_action_trajectories(trajectories, config["path_trajectory"])

trajectories = load_action_trajectories(config["path_trajectory"])
trajectories_sas = convert_traj_to_sas(trajectories)

exp_rep_dist = ErDist(max_n_trajectories=10000, trajectories_list=trajectories_sas)

dist_encoder_model = MlpDistEncoder(config["in_d"],
                                    config["out_d"],
                                    config["dist_type"],
                                    config["in_dist_d"],
                                    config["out_dist_d"])

optimizer = torch.optim.AdamW(dist_encoder_model.parameters(), weight_decay=0, lr=config["l_rate"], amsgrad=config["amsgrad"])

# train the dist model
for step in trange(config["gradient_steps"], desc="Steps of gradient", leave=True):

    dist_encoder_model.train()
    loss_o, loss_c, loss = dist_encoder_model.training_step(exp_rep_dist, optimizer, config)

    wandb.log({"loss_o": loss_o, "loss_c": loss_c, "loss": loss})

# train the action model
action_encoder_model = ActionEncoder(config["in_a_d"], config["out_d"])

optimizer = torch.optim.AdamW(action_encoder_model.parameters(), weight_decay=0, lr=config["l_rate_action"], amsgrad=config["amsgrad"])

for step in trange(config["gradient_steps_action"], desc="Steps of gradient", leave=True):

    loss = action_encoder_model.training_step(dist_encoder_model, exp_rep_dist, optimizer, config)

    wandb.log({"loss_action": loss})

# # test the action model
#
# action_encoder_model.eval()
# dist_encoder_model.eval()
#
# s1, s2, a = exp_rep_dist.get_batch_action(batch_size=config["batch_size_action"])
#
# z1 = dist_encoder_model(s1)
# z2 = dist_encoder_model(s2)
#
# z1_a = action_encoder_model(z1, a)
#
# for z1, z2, a, z1_a in zip(z1, z2, a, z1_a):
#     print(z1, z2, a, z1_a, dist_encoder_model.dist(z1_a, z2))

full_model = LearnedModel(dist_encoder_model, action_encoder_model)
#
# save the full model
torch.save(full_model, config["model_path"])


