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
from lambda_schedulers import *

# configurations
env = gym.make("MountainCar-v0")
observe_dim = env.observation_space.shape[0]
action_num = env.action_space.n
model = LearnedModel(2, 64, action_num)
model.load_state_dict(
    torch.load("/home/lorenzo/Documenti/UPF/MDP_temporal_representation/Models/MountainCar_action_states_dictionary"))
model.eval()
z_goal_states = []
z_goal_states.append(model.encode_state(torch.FloatTensor([[0.50427865, 0.02712902]])))
z_goal_states.append(model.encode_state(torch.FloatTensor([[0.51197713, 0.02856222]])))
z_goal_states.append(model.encode_state(torch.FloatTensor([[0.52146404, 0.02956875]])))
max_episodes = 1000
max_steps = 200
solved_reward = 200
solved_repeat = 5

# Fixing all seed for reproducibility
seed = 1
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
env.seed(seed)

# wandb_group = "MountainCar_DQN_Reshaped"
# wandb_name = "MountainCar_DQN_Reshaped_seed_" + str(seed)
#
# run = wandb.init(project='MDP_TR', entity='lsteccanella',
#                  group=wandb_group, settings=wandb.Settings(start_method="fork"))
# wandb.run.name = wandb_name

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

# model definition
class QNet(torch.nn.Module):
    def __init__(self, state_dim, action_num):
        super().__init__()

        self.fc1 = torch.nn.Linear(state_dim, 16)
        self.fc2 = torch.nn.Linear(16, 16)
        self.fc3 = torch.nn.Linear(16, action_num)

    def forward(self, some_state):
        a = torch.selu(self.fc1(some_state))
        a = torch.selu(self.fc2(a))
        return self.fc3(a)

q_net = QNet(observe_dim, action_num)
q_net_t = QNet(observe_dim, action_num)

dqn = DQN(q_net, q_net_t, torch.optim.Adam, torch.nn.MSELoss(reduction="sum"), epsilon_decay=0.999, discount=0.95,
          learning_rate=0.01, update_rate=0.1)

episode, step, reward_fulfilled = 0, 0, 0
smoothed_total_reward = -200
tot_steps = 0
lambda_s = LinearLS(0, n_epochs=100)

while True:
    episode += 1
    total_reward = 0
    total_o_reward = 0
    total_r_reward = 0
    terminal = False
    step = 0
    state = env.reset(); s=state
    state = torch.tensor(state, dtype=torch.float32).view(1, observe_dim)
    episode_transitions = []
    lambda_w = 0.5

    while not terminal and step <= max_steps:
        step += 1
        tot_steps += 1
        with torch.no_grad():
            old_state = state
            # agent model inference
            action = dqn.act_discrete_with_noise({"some_state": old_state})
            state, reward, terminal, _ = env.step(action.item()); s_ = state
            o_reward = reward
            s = torch.FloatTensor([state])
            z = model.encode_state(s)
            dist = min_dist_to_goal_states(model, z, z_goal_states)
            fake_reward = -0.01 * dist
            state = torch.tensor(state, dtype=torch.float32).view(1, observe_dim)
            total_reward += o_reward
            total_o_reward += o_reward
            total_r_reward += fake_reward

            s = s_
            episode_transitions.append(
                {
                    "state": {"some_state": old_state},
                    "action": {"action": action},
                    "next_state": {"some_state": state},
                    "reward": o_reward,
                    "terminal": terminal or step == max_steps,
                }
            )
    # wandb.log({"reward": total_o_reward,
    #            "reshaped_reward": total_r_reward,
    #            "epsilon_decay": dqn.epsilon},
    #           step=tot_steps)

    dqn.store_episode(episode_transitions)
    env.render()
    # update, update more if episode is longer, else less
    for _ in range(step):
        dqn.update_w_heuristic(lambda_s(), z_goal_states, model, -0.01)

    lambda_s.update()

    # show reward
    smoothed_total_reward = smoothed_total_reward * 0.9 + total_reward * 0.1
    logger.info(f"Episode {episode} total reward={smoothed_total_reward:.2f} fake reward={total_r_reward:.2f}")
