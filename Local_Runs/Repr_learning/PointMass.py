from tqdm import trange
from Utils.Utils import *
import os
from Models.Models import *
dirname = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

config= {
    "seed": 0,
    "path_trajectory": dirname + "/Data/trajectory_PointMass.npy",
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
    "gradient_steps": 100,
    "in_a_d": 5,
    "l_rate_action": 0.0005,
    "amsgrad_action": True,
    "batch_size_action": 128,
    "gradient_steps_action": 10000
}

np.random.seed(config["seed"])
torch.manual_seed(config["seed"])
random.seed(config["seed"])

env = PointmassEnv(max_n_steps_episode=50)
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
s1_print, s2_print, _ = exp_rep_dist.get_batch(batch_size=1024, d_tresh=None)
dist_pred = dist_encoder_model.dist(dist_encoder_model(s1_print), dist_encoder_model(s2_print)).detach().numpy()

sum_e = 0
for s1, s2, pred_d in zip(s1_print, s2_print, dist_pred):
    true_d = dist_matrix_original[tuple(s1.detach().numpy())][tuple(s2.detach().numpy())]
    sum_e += (pred_d - true_d) ** 2
print("MSE", sum_e / len(dist_pred))

# # train the action model
# action_encoder_model = ActionEncoder(config["in_a_d"], config["out_d"])
#
# optimizer = torch.optim.AdamW(action_encoder_model.parameters(), weight_decay=0, lr=config["l_rate_action"], amsgrad=config["amsgrad"])
#
# for step in trange(config["gradient_steps_action"], desc="Steps of gradient", leave=True):
#
#     loss = action_encoder_model.training_step(dist_encoder_model, exp_rep_dist, optimizer, config)
#
# print("Loss action embedding, loss", loss)
#
# # # test the action model
# #
# # action_encoder_model.eval()
# # dist_encoder_model.eval()
# #
# # s1, s2, a = exp_rep_dist.get_batch_action(batch_size=config["batch_size_action"])
# #
# # z1 = dist_encoder_model(s1)
# # z2 = dist_encoder_model(s2)
# #
# # z1_a = action_encoder_model(z1, a)
# #
# # for z1, z2, a, z1_a in zip(z1, z2, a, z1_a):
# #     print(z1, z2, a, z1_a, dist_encoder_model.dist(z1_a, z2))
#
# full_model = LearnedModel(dist_encoder_model, action_encoder_model)
#
# # save the full model
# torch.save(full_model, dirname + "/Saved_models_v2/PointMass_full_model")




