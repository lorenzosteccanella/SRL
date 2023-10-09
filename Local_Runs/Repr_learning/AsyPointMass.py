from tqdm import trange
from Utils.Utils import *
import os
from Models.Models import *
from Envs.PointmassEnv import AsyPointmassEnv
dirname = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

config= {
    "seed": 0,
    "path_trajectory": dirname + "/Data/trajectory_AsyPointMass.npy",
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
    "dist_discount": 0.97,
    "max_dist_con": 4,
    "weight_constrains": 10,
    "gradient_steps": 100000,
    "in_a_d": 5,
    "l_rate_action": 0.0005,
    "amsgrad_action": True,
    "batch_size_action": 128,
    "gradient_steps_action": 10000
}

np.random.seed(config["seed"])
torch.manual_seed(config["seed"])
random.seed(config["seed"])

env = AsyPointmassEnv(max_n_steps_episode=50)
trajectories, _ = collect_action_trajectories(env, 100, False, float("inf"))
save_action_trajectories(trajectories, config["path_trajectory"])

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

# # test the dist model

dist_matrix_original = env.compute_MAD_distance()

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
print("MSE", sum_e / len(dist_pred))









