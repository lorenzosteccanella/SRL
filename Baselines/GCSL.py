from collections import deque

import numpy as np
import torch
from torch import nn

class GCSL_NN(nn.Module):
    def __init__(self, max_n_steps_episode, input_size, actions):
        super().__init__()
        self.max_n_steps_episode = max_n_steps_episode
        self.actions = actions
        self.l1 = nn.Linear(input_size, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, len(actions))

        torch.nn.init.kaiming_normal_(self.l1.weight.data, mode='fan_in', nonlinearity='selu')
        torch.nn.init.kaiming_normal_(self.l2.weight.data, mode='fan_in', nonlinearity='selu')
        torch.nn.init.kaiming_normal_(self.l3.weight.data, mode='fan_in', nonlinearity='linear')

        self.activation = torch.nn.SELU()

    def net(self, x):
        x = self.l1(x)
        x = self.activation(x)
        x = self.l2(x)
        x = self.activation(x)
        x = self.l3(x)
        return x

    def forward(self, state, goal, horizon):
        if horizon is None:
            x = torch.cat([state, goal], -1)
        else:
            horizon = horizon / self.max_n_steps_episode  # Normalize between [0, 1]
            x = torch.cat([state, goal, horizon], -1)
        logits = self.net(x)
        return logits

    def get_action(self, state, goal, horizon):
        # Put into PyTorch Notation
        state_torch = torch.tensor(state, dtype=torch.float32)[None]
        goal_torch = torch.tensor(goal, dtype=torch.float32)[None]
        if horizon is not None:
            horizon_torch = torch.tensor(horizon, dtype=torch.float32)[None, None]
            logits_torch = self.forward(state_torch, goal_torch, horizon_torch)[0]
        else:
            logits_torch = self.forward(state_torch, goal_torch, None)[0]
        probabilities_torch = torch.softmax(logits_torch, -1)
        probabilities = probabilities_torch.detach().numpy()

        r_a_i = np.random.choice(len(self.actions), p=probabilities)
        return self.actions[r_a_i]


class GCSL_experience_replay():

    def __init__(self, size=20000):
        self.trajectories = deque(maxlen=size)

    def add_trajectory(self, states, actions, desired_goal):
        traj = {
            'states': np.array(states),
            'actions': np.array(actions),
            'desired_goal': desired_goal,
        }
        self.trajectories.append(traj)

    def load_trajectories(self, trajectories, desired_goal, discretize_action=False, actions_list=None):
        if not discretize_action:
            for trajectory in trajectories:
                states = []
                actions = []
                for state, one_hot_a in trajectory:
                    states.append(state)
                    actions.append(np.argmax(one_hot_a))
                self.add_trajectory(states, actions, desired_goal)

        if discretize_action:
            assert actions_list is not None
            for trajectory in trajectories:
                states = []
                actions = []
                for state, a in trajectory:
                    def find_nearest(array, value):
                        array = np.asarray(array)
                        idx = (np.abs(array - value).sum(axis=1)).argmin()
                        return idx
                    states.append(state)
                    actions.append(find_nearest(actions_list, a))
                self.add_trajectory(states, actions, desired_goal)

    def sample_batch(self, batch_size):
        # get data:
        s_batch = []
        g_batch = []
        h_batch = []
        a_batch = []
        for sample in range(batch_size):
            # Sample a trajectory and timesteps
            traj_i = np.random.choice(len(self.trajectories))
            trajectory = self.trajectories[traj_i]
            t1, t2 = np.random.randint(0, len(trajectory['states'])-1, size=2)
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

        return s_batch, g_batch, h_batch, a_batch
    

class GCSL:

    def __init__(self, model, optimizer, learning_rate):
        self.model = model
        self.optimizer_func = optimizer
        self.optimizer = self.optimizer_func(self.model.parameters(), lr=learning_rate)

    def train(self, s_batch, g_batch, h_batch, a_batch):
        # Optimize agent(s, g, h) to imitate action a
        self.optimizer.zero_grad()
        if h_batch is not None:
            x = self.model(torch.stack(list(s_batch)), torch.stack(list(g_batch)), torch.stack(list(h_batch)))
        else:
            x = self.model(torch.stack(list(s_batch)), torch.stack(list(g_batch)), None)
        y = torch.stack(list(a_batch))
        loss = nn.functional.cross_entropy(x, y)
        loss.backward()
        self.optimizer.step()

    def act(self, state, goal, horizon):
        return self.model.get_action(state, goal, horizon)
