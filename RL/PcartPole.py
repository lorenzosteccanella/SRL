from machin.frame.algorithms import DQN
from machin.utils.logging import default_logger as logger
from torch.distributions import Categorical
import torch
import torch.nn as nn
import gym
import numpy as np
import random
from Problem_formulation.minibatch_deep_d_PER import LearnedModel
import math
import wandb
from lambda_schedulers import TanhLS


# configurations
env = gym.make("CartPole-v0")
observe_dim = env.observation_space.shape[0]
action_num = env.action_space.n
model = LearnedModel(4, 64, action_num)
model.load_state_dict(
    torch.load("/home/lorenzo/Documenti/UPF/MDP_temporal_representation/Models/CartPole_action_states_dictionary"))
model.eval()
z_goal_states = []
for pos in range(-10, +10+1, 1):
    z_goal_states.append(model.encode_state(torch.FloatTensor([[0.01 * pos, 0,  0, 0]])))
max_episodes = 1000
max_steps = 200
max_tot_steps = 20000
solved_reward = 200
solved_repeat = 5

# Fixing all seed for reproducibility
seed = 5
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
env.seed(seed)

wandb_group = "CartPole_DQN_H"
wandb_name = "CartPole_DQN_H_seed_" + str(seed)

run = wandb.init(project='MDP_DIST', entity='lsteccanella',
                 group=wandb_group, settings=wandb.Settings(start_method="fork"))
wandb.run.name = wandb_name

# model definition
class QNet(torch.nn.Module):
    def __init__(self, state_dim, action_num):
        super().__init__()

        self.fc1 = torch.nn.Linear(state_dim, 16)
        self.fc2 = torch.nn.Linear(16, 16)
        self.fc3 = torch.nn.Linear(16, action_num)

    def forward(self, some_state):
        a = torch.relu(self.fc1(some_state))
        a = torch.relu(self.fc2(a))
        return self.fc3(a)

def min_dist_to_goal_states(model, z, z_goal_states):
    tmp_dist = float("inf")
    for z_goal_state in z_goal_states:
        z_dist = model.encoder.dist(z_goal_state, z).detach().numpy()[0]
        if tmp_dist > z_dist:
            tmp_dist = z_dist
    return tmp_dist

def avg_dist_to_goal_states(model, z, z_goal_states):
    tmp_dist = 0
    for z_goal_state in z_goal_states:
        tmp_dist += model.encoder.dist(z_goal_state, z).detach().numpy()[0]
    return tmp_dist/len(z_goal_states)

q_net = QNet(observe_dim, action_num)
q_net_t = QNet(observe_dim, action_num)
discount = 0.95
dqn = DQN(q_net, q_net_t, torch.optim.Adam, torch.nn.MSELoss(reduction="sum"), epsilon_decay=0.999, discount=discount,
          learning_rate=0.001, update_rate=0.1)

episode, step, reward_fulfilled = 0, 0, 0
smoothed_total_reward = 0
tot_steps = 0

while tot_steps < max_tot_steps:
    episode += 1
    total_reward = 0
    total_o_reward = 0
    total_r_reward = 0
    terminal = False
    step = 0
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32).view(1, observe_dim)
    episode_transitions = []

    while not terminal and step <= max_steps:
        step += 1
        tot_steps += 1
        with torch.no_grad():
            old_state = state
            # agent model inference
            action = dqn.act_discrete_with_noise({"some_state": old_state})
            state, reward, terminal, _ = env.step(action.item())
            o_reward = reward
            state = torch.tensor(state, dtype=torch.float32).view(1, observe_dim)
            z = model.encode_state(old_state)
            z_ = model.encode_state(state)
            dist_z = min_dist_to_goal_states(model, z, z_goal_states)
            dist_z_ = min_dist_to_goal_states(model, z_, z_goal_states)
            fake_reward = reward - (discount * (dist_z_ - dist_z))
            total_reward += o_reward
            total_o_reward += o_reward
            total_r_reward += fake_reward
            episode_transitions.append(
                {
                    "state": {"some_state": old_state},
                    "action": {"action": action},
                    "next_state": {"some_state": state},
                    "reward": fake_reward,
                    "terminal": terminal or step == max_steps,
                }
            )
    wandb.log({"reward": total_o_reward,
               "reshaped_reward": total_r_reward,
               "epsilon_decay": dqn.epsilon},
              step=tot_steps)
    dqn.store_episode(episode_transitions)

    # update, update more if episode is longer, else less
    for _ in range(step):
        dqn.update()

    # show reward
    smoothed_total_reward = smoothed_total_reward * 0.9 + total_reward * 0.1
    logger.info(f"Episode {episode} total reward={smoothed_total_reward:.2f} fake reward={total_r_reward:.2f} epsilon_dqn={dqn.epsilon:.2f}")
