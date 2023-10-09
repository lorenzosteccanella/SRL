import sys
import os

dirpath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(dirpath)

from tqdm import trange
from Utils.Utils import *
from Models.Models import *
from Envs.PointmassEnv import PointmassEnv

seed = int(sys.argv[1])

config= {
    "seed": seed,
    "path_trajectory": dirpath + "/Data/trajectory_PointMass.npy",
    "in_d": 2,
    "out_d": 2,
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
    "in_a_d": 5,
    "l_rate_action": 0.0005,
    "amsgrad_action": True,
    "batch_size_action": 128,
    "gradient_steps_action": 10000,

    "wandb_group": "PointMass_WN",

    "model_path": dirpath + "/Saved_models/PointMass_WN_" + str(seed)
}

np.random.seed(config["seed"])
torch.manual_seed(config["seed"])
random.seed(config["seed"])

run = wandb.init(project='MAD_SemiNorm', entity='lsteccanella', config=config,
                 group=config["wandb_group"], settings=wandb.Settings(start_method="fork"))
wandb.run.name = "repr_learning_seed_" + str(config["seed"])

env = PointmassEnv(max_n_steps_episode=50)
# trajectories, _ = collect_action_trajectories(env, 100, False, float("inf"))
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
dist_matrix_original = env.compute_MAD_distance()

# train the dist model
for step in trange(config["gradient_steps"], desc="Steps of gradient", leave=True):

    dist_encoder_model.train()
    loss_o, loss_c, loss = dist_encoder_model.training_step(exp_rep_dist, optimizer, config)

    wandb.log({"loss_o": loss_o, "loss_c": loss_c, "loss": loss})

    if step % 100 == 0:
        # # test the dist model
        dist_encoder_model.eval()
        # s1_print, s2_print, _ = exp_rep_dist.get_batch(batch_size=1024, d_tresh=None)
        all_states = [(i, j) for i in range(int(env.state_space.high[0])) for j in range(int(env.state_space.high[0]))]
        all_pairs_states = np.array([(i, j) for i in all_states for j in all_states])
        all_s1 = torch.from_numpy(all_pairs_states[:, 0]).float()
        all_s2 = torch.from_numpy(all_pairs_states[:, 1]).float()
        dist_pred = dist_encoder_model.dist(dist_encoder_model(all_s1), dist_encoder_model(all_s2)).detach().numpy()

        sum_e = 0
        for all_s1, s2, pred_d in zip(all_s1, all_s2, dist_pred):
            true_d = dist_matrix_original[tuple(all_s1.detach().numpy())][tuple(s2.detach().numpy())]
            sum_e += (pred_d - true_d) ** 2

        wandb.log({"MSE": sum_e / len(dist_pred)})

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