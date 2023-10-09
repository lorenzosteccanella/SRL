from tqdm import trange

from Envs.RingEnv import RingEnv
from ExpReplay.ErDist import ErDist
from Models.Models import MlpDistEncoder
from Utils.Utils import *

config= {
    "seed": 0,
    "in_d": 3,
    "out_d": 8,
    "in_dist_d": 8,
    "out_dist_d": 8,
    "dist_type": "L1",
    "l_rate": 0.0005,
    "amsgrad": True,
    "batch_size_o": 64,
    "batch_size_c": 512,
    "max_dist_obj": None,
    "dist_discount": 0.97,
    "max_dist_con": 4,
    "weight_constrains": 10,
    "gradient_steps": 10000
}

np.random.seed(config["seed"])
torch.manual_seed(config["seed"])
random.seed(config["seed"])

env = RingEnv(max_n_steps_episode=50)
trajectories, states = collect_action_trajectories(env, 100, False)

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

# all combination of states
states_comb = np.array(list(itertools.product(states, states)))

# test the dist model
dist_encoder_model.eval()
s1 = torch.FloatTensor(states_comb[:, 0]) #torch.FloatTensor(np.array([[1, 0, 0], [0, 1, 0]]))
s2 = torch.FloatTensor(states_comb[:, 1]) #torch.FloatTensor(np.array([[0, 1, 0], [1, 0, 0]]))

z1 = dist_encoder_model(s1)
z2 = dist_encoder_model(s2)

for s1, s2, pred_d in zip(s1, s2, dist_encoder_model.dist(z1, z2)):
    print("s1", s1, "s2", s2, "pred_d", pred_d)







