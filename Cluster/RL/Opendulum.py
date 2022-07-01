import os
import sys
dirpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(dirpath))

from machin.frame.algorithms import DDPG
from machin.utils.logging import default_logger as logger
import torch
import gym
import numpy as np
import random
import wandb

# configurations
env = gym.make("Pendulum-v0")
observe_dim = env.observation_space.shape[0]
action_dim = 1
max_episodes = 1000
action_range = 2
max_steps = 200
noise_param = (0, 0.2)
noise_mode = "normal"
max_tot_steps = 100000

# Fixing all seed for reproducibility
seed = int(sys.argv[1])
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
env.seed(seed)

discount = float(sys.argv[2])
epsilon = 0

wandb_group = "Pendulum_DQN_O" + "_" + str(discount) + "_" + str(epsilon)
wandb_name = "Pendulum_DQN_O_seed_" + str(seed) + "_" + str(discount) + "_" + str(epsilon)

run = wandb.init(project='MDP_DIST', entity='lsteccanella',
                 group=wandb_group, settings=wandb.Settings(start_method="fork"))
wandb.run.name = wandb_name

# model definition
class Actor(torch.nn.Module):
    def __init__(self, state_dim, action_dim, action_range):
        super().__init__()

        self.fc1 = torch.nn.Linear(state_dim, 32)
        self.fc2 = torch.nn.Linear(32, 32)
        self.fc3 = torch.nn.Linear(32, action_dim)
        self.action_range = action_range

        torch.nn.init.kaiming_normal_(self.fc1.weight.data, mode='fan_in', nonlinearity='selu')
        torch.nn.init.kaiming_normal_(self.fc2.weight.data, mode='fan_in', nonlinearity='selu')
        torch.nn.init.kaiming_normal_(self.fc3.weight.data, mode='fan_in', nonlinearity='linear')

        self.activation = torch.nn.SELU()

    def forward(self, state):
        a = self.activation(self.fc1(state))
        a = self.activation(self.fc2(a))
        a = torch.tanh(self.fc3(a)) * self.action_range
        return a


class Critic(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.fc1 = torch.nn.Linear(state_dim + action_dim, 32)
        self.fc2 = torch.nn.Linear(32, 32)
        self.fc3 = torch.nn.Linear(32, 1)

        torch.nn.init.kaiming_normal_(self.fc1.weight.data, mode='fan_in', nonlinearity='selu')
        torch.nn.init.kaiming_normal_(self.fc2.weight.data, mode='fan_in', nonlinearity='selu')
        torch.nn.init.kaiming_normal_(self.fc3.weight.data, mode='fan_in', nonlinearity='linear')

        self.activation = torch.nn.SELU()

    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)
        q = self.activation(self.fc1(state_action))
        q = self.activation(self.fc2(q))
        q = self.fc3(q)
        return q


actor = Actor(observe_dim, action_dim, action_range)
actor_t = Actor(observe_dim, action_dim, action_range)
critic = Critic(observe_dim, action_dim)
critic_t = Critic(observe_dim, action_dim)

ddpg = DDPG(
    actor, actor_t, critic, critic_t, torch.optim.Adam, torch.nn.MSELoss(reduction="sum"), discount=discount,
)

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
    state = env.reset();
    state = torch.tensor(state, dtype=torch.float32).view(1, observe_dim)
    episode_transitions = []

    while not terminal and step <= max_steps:
        step += 1
        tot_steps += 1
        with torch.no_grad():
            old_state = state
            # agent model inference
            action = ddpg.act_with_noise(
                {"state": old_state}, noise_param=noise_param, mode=noise_mode
            )
            state, reward, terminal, _ = env.step(action.numpy());
            reward = reward[0]
            o_reward = reward
            state = torch.tensor(state, dtype=torch.float32).view(1, observe_dim)
            fake_reward = reward
            total_reward += o_reward
            total_o_reward += o_reward
            total_r_reward += fake_reward

            #env.render()
            episode_transitions.append(
                {
                    "state": {"state": old_state},
                    "action": {"action": action},
                    "next_state": {"state": state},
                    "reward": reward,
                    "terminal": terminal or step == max_steps,
                }
            )

    wandb.log({"reward": total_o_reward,
               "reshaped_reward": total_r_reward},
              step=tot_steps)

    ddpg.store_episode(episode_transitions)

    # update, update more if episode is longer, else less
    for _ in range(step):
        ddpg.update()

    # show reward
    smoothed_total_reward = smoothed_total_reward * 0.9 + total_reward * 0.1
    logger.info(f"Episode {episode} total reward={smoothed_total_reward:.2f} total o reward={total_o_reward:.2f} fake reward={total_r_reward:.2f}")
