from tqdm import trange
from Utils.Utils import *
import os
from Models.Models import *
dirname = os.path.dirname(os.path.dirname(__file__))

config= {
    "seed": 0,
    "path_trajectory": dirname + "/Data/trajectory_CartPole.npy",
    "in_d": 4,
    "out_d": 16,
    "dist_type": "WideNorm",
    "in_dist_d": 8,
    "out_dist_d": 8,
    "l_rate": 0.0005,
    "amsgrad": True,
    "batch_size_o": 64,
    "batch_size_c": 512,
    "max_dist_obj": None,
    "dist_discount": 0.97,
    "max_dist_con": 10,
    "weight_constrains": 10,
    "gradient_steps": 500000,
    "in_a_d": 2,
    "l_rate_action": 0.0005,
    "amsgrad_action": True,
    "batch_size_action": 128,
    "gradient_steps_action": 500000
}

np.random.seed(config["seed"])
torch.manual_seed(config["seed"])
random.seed(config["seed"])

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
    loss_o, loss_c, loss = dist_encoder_model.training_step(exp_rep_dist, optimizer, config)

print("Loss distance embedding, loss_o", loss_o, "loss_c", loss_c, "loss", loss, "\n")

# train the action model
action_encoder_model = ActionEncoder(config["in_a_d"], config["out_d"])

optimizer = torch.optim.AdamW(action_encoder_model.parameters(), weight_decay=0, lr=config["l_rate_action"], amsgrad=config["amsgrad"])

for step in trange(config["gradient_steps_action"], desc="Steps of gradient", leave=True):

    loss = action_encoder_model.training_step(dist_encoder_model, exp_rep_dist, optimizer, config)

print("Loss action embedding, loss", loss)

full_model = LearnedModel(dist_encoder_model, action_encoder_model)

# save the full model
torch.save(full_model, dirname + "/Saved_models_v2/CartPole_full_model")