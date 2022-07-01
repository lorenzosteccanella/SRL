import os
import sys
dirpath = os.path.dirname(os.path.abspath(__file__))
dirname = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(dirpath))

from collections import deque
from Envs.PointmassEnv import PointmassEnv
import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
import wandb

class NNAgent(nn.Module):
    def __init__(self, max_n_steps_episode):
        super().__init__()
        self.max_n_steps_episode = max_n_steps_episode
        self.net = nn.Sequential(
            nn.Linear(5, 128),  # Input 2 (state) + 2 (goal) + 1 (horizon)
            nn.SELU(),
            nn.Linear(128, 128),
            nn.SELU(),
            nn.Linear(128, 5),  # Output: 5 actions
        )

    def forward(self, state, goal, horizon):
        horizon = horizon / self.max_n_steps_episode  # Normalize between [0, 1]
        x = torch.cat([state, goal, horizon], -1)
        logits = self.net(x)
        return logits

    def get_action(self, state, goal, horizon):
        # Put into PyTorch Notation
        state_torch = torch.tensor(state, dtype=torch.float32)[None]
        goal_torch = torch.tensor(goal, dtype=torch.float32)[None]
        horizon_torch = torch.tensor(horizon, dtype=torch.float32)[None, None]
        logits_torch = self.forward(state_torch, goal_torch, horizon_torch)[0]
        probabilities_torch = torch.softmax(logits_torch, -1)
        probabilities = probabilities_torch.detach().numpy()

        return np.random.choice(5, p=probabilities)

def sample_trajectory(env, agent, T=200):
    # Sample a target goal (fixed for episode)
    desired_goal = env.sample_goal()

    # Default control loop
    state = env.reset()
    states = []
    actions = []
    for i in range(T):
        states.append(state)

        action = agent.get_action(state=state,
                                  goal=desired_goal,
                                  horizon=np.array(T-i, dtype=float))
        actions.append(action)

        state, _, _, _ = env.step(action)

    return {
      'states': np.array(states),
      'actions': np.array(actions),
      'desired_goal': desired_goal,
    }

def sample_random_trajectory(env, agent, T=200):
    # Sample a target goal (fixed for episode)
    desired_goal = env.sample_goal()

    # Default control loop
    state = env.reset()
    states = []
    actions = []
    for i in range(T):
        states.append(state)

        action = np.random.choice(env.action_space.n)
        actions.append(action)

        state, _, _, _ = env.step(action)

    return {
      'states': np.array(states),
      'actions': np.array(actions),
      'desired_goal': desired_goal,
    }

def evaluate_agent(env, agent, n=50, T=200):
    distances = []
    len_path = []
    for _ in range(n):
        trajectory = sample_trajectory(env, agent, T)
        distances.append(np.linalg.norm(trajectory['states'][-1] - trajectory['desired_goal']))
        dist=0
        for i in range(len(trajectory)-1):
            dist += np.linalg.norm(trajectory['states'][i] - trajectory['desired_goal'])
        len_path.append(dist)

    return distances, len_path

def plot_trajectory(trajectory, ax=None):
    if ax is None:
        ax = plt.gca()
    # Draw path
    ax.plot(*trajectory['states'].T)
    # Draw goal
    ax.scatter(0, 0, s=200, marker='s')
    ax.scatter(trajectory['desired_goal'][0], trajectory['desired_goal'][1], s=400, marker='*')
    # Draw boundary
    ax.vlines([-1, 1], -1, 1)
    ax.hlines([-1, 1], -1, 1)

    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.axis('off')

def visualize_agent(env, agent):
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    for ax in axes:
        plot_trajectory(sample_trajectory(env, agent), ax=ax)

seed = 1
np.random.seed(seed)
torch.manual_seed(seed)

wandb_group = "GCSL_PointMass"
wandb_name = "GCSL_PointMass_seed_" + str(seed)

run = wandb.init(project='MDP_DIST', entity='lsteccanella',
                 group=wandb_group, settings=wandb.Settings(start_method="fork"))
wandb.run.name = wandb_name

env = PointmassEnv()
env.seed(seed)

buffer = []
n_episodes = 100
n_steps_per_episode = 10000
max_n_steps_episode = 50


batch_size = 64
s_batch = deque(maxlen=batch_size)
g_batch = deque(maxlen=batch_size)
h_batch = deque(maxlen=batch_size)
a_batch = deque(maxlen=batch_size)

agent = NNAgent(max_n_steps_episode)
learning_rate = 1e-4
optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate)

for episode in range(1, n_episodes):
    # agent = NNAgent(max_n_steps_episode)
    # learning_rate = 1e-4
    # optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate)
    # Collect more data and put it in the replay buffer
    new_trajectory = sample_random_trajectory(env, agent, max_n_steps_episode)
    buffer.append(new_trajectory)

# GCSL optimization
for step in range(n_steps_per_episode):

    for sample in range(batch_size):
        # Sample a trajectory and timesteps
        trajectory = buffer[np.random.choice(len(buffer))]
        t1, t2 = np.random.randint(0, max_n_steps_episode, size=2)
        t1, t2 = min([t1, t2]), max([t1, t2])

        # Create optimal ((s, g, h), a) data
        s = trajectory['states'][t1]
        g = trajectory['states'][t2]
        h = t2 - t1
        a = trajectory['actions'][t1]
        s = torch.tensor(s, dtype=torch.float32)
        g = torch.tensor(g, dtype=torch.float32)
        h = torch.tensor(h, dtype=torch.float32)[None]
        a = torch.tensor(a)
        s_batch.append(s)
        g_batch.append(g)
        h_batch.append(h)
        a_batch.append(a)

    # Optimize agent(s, g, h) to imitate action a
    optimizer.zero_grad()
    x = agent(torch.stack(list(s_batch)), torch.stack(list(g_batch)), torch.stack(list(h_batch)))
    y = torch.stack(list(a_batch))
    loss = nn.functional.cross_entropy(x, y)
    loss.backward()
    optimizer.step()

# Print agent performance once in a while
print('##### Episode %d #####'%episode)
distances, len_path = evaluate_agent(env, agent, 200, max_n_steps_episode)

print('Mean Distance to Goal:  %.3f' % np.mean(distances))
print('Min Distance to Goal:  %.3f' % np.min(distances))
print('Max Distance to Goal:  %.3f' % np.max(distances))
print('Mean Distance Traversed to Goal: %.3f' % np.mean(len_path))

wandb.log({"Mean Distance to Goal": np.mean(distances),
           "Min Distance to Goal": np.min(distances),
           "Max Distance to Goal:": np.max(distances),
           "Mean Distance Traversed to Goal": np.mean(len_path),
           "n_trajectories": episode},
          step=episode)

visualize_agent(env, agent)
