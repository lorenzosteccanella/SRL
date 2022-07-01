import itertools
import random
import time
import numpy as np
import torch
import wandb

class Planner():

    def __init__(self, model, actions, max_n_rand_traj, max_horizon):
        self.model = model
        self.actions = actions
        self.max_n_rand_traj = max_n_rand_traj
        self.max_horizon = max_horizon

    def get_z_goal_states(self, goal_states):
        z_goal_states = []
        for goal_state in goal_states:
            z_goal_states.append(self.model.encode_state(torch.FloatTensor([goal_state])))

        return z_goal_states

    def min_dist_to_goal_states(self, z, z_goal_states):
        tmp_dist = float("inf")
        for z_goal_state in z_goal_states:
            z_dist = self.model.encoder.dist(z_goal_state, z).detach().numpy()[0]
            if tmp_dist > z_dist:
                tmp_dist = z_dist
        return tmp_dist

    def avg_dist_to_goal_states(self, z, z_goal_states):
        tmp_dist = 0
        for z_goal_state in z_goal_states:
            tmp_dist += self.model.encoder.dist(z_goal_state, z).detach().numpy()[0]
        return tmp_dist/len(z_goal_states)

    def one_hot_action(self, a):
        one_hot_a = np.zeros(len(self.actions))
        one_hot_a[a] = 1
        one_hot_a = np.expand_dims(one_hot_a, axis=0)
        one_hot_a = torch.FloatTensor(one_hot_a)
        return one_hot_a

    def action_step_lookahead(self, z, action_traj, dist, z_goal_states):
        for a, step in zip(action_traj, range(self.max_horizon)):
            one_hot_a = self.one_hot_action(a)
            z_ = self.model.encode_action(z, one_hot_a)
            # if step == horizon-1:
            #     dist = min_dist_to_goal_states(model, z_, z_goal_states)
            dist += self.min_dist_to_goal_states(z_, z_goal_states)
            z = z_
        return dist

    def cont_action_step_lookahead(self, z, action_traj, dist, z_goal_states):
        for a, step in zip(action_traj, range(self.max_horizon)):
            z_ = self.model.encode_action(z, torch.FloatTensor([a]))
            # if step == horizon-1:
            #     dist = min_dist_to_goal_states(model, z_, z_goal_states)
            dist += self.min_dist_to_goal_states(z_, z_goal_states)
            z = z_
        return dist

    def multi_step_lookahead(self, s, z_goal_states):
        horizon = self.max_horizon
        action_traj_comb = list(itertools.combinations_with_replacement(self.actions, horizon))
        action_traj_comb = random.sample(action_traj_comb, min(len(action_traj_comb), self.max_n_rand_traj))
        actions_values = np.ones(len(self.actions)) * float("inf")
        s = torch.FloatTensor(np.array([s]))
        for a in self.actions:
            one_hot_a = self.one_hot_action(a)
            z_ = self.model.encode_next_state(s, one_hot_a)
            dist = self.min_dist_to_goal_states(z_, z_goal_states)
            for action_traj in action_traj_comb:
                assert len(action_traj) > 0
                dist = self.action_step_lookahead(z_, action_traj, dist, z_goal_states)
                actions_values[a] = min(dist, actions_values[a])#(dist/(len(action_traj_comb)+1))

        return np.argmin(actions_values)

    def prob_multi_step_lookahead(self, s, z_goal_states, temperature=1, verbose=False):
        horizon = self.max_horizon
        action_traj_comb = list(itertools.combinations_with_replacement(self.actions, horizon))
        action_traj_comb = random.sample(action_traj_comb, min(len(action_traj_comb), self.max_n_rand_traj))
        actions_values = np.ones(len(self.actions)) * float("inf")
        s = torch.FloatTensor(np.array([s]))
        for a in self.actions:
            one_hot_a = self.one_hot_action(a)
            z_ = self.model.encode_next_state(s, one_hot_a)
            dist = self.min_dist_to_goal_states(z_, z_goal_states)
            for action_traj in action_traj_comb:
                assert len(action_traj) > 0
                dist = self.action_step_lookahead(z_, action_traj, dist, z_goal_states)
                actions_values[a] = min(dist, actions_values[a])#(dist/(len(action_traj_comb)+1))
        m = torch.nn.Softmin(dim=0)
        softmin_p = m(torch.FloatTensor(actions_values/temperature)).detach().numpy()
        if verbose: print(softmin_p)
        return np.random.choice(self.actions, p=softmin_p)

    def cont_multi_step_lookahead(self, s, z_goal_states):
        horizon = self.max_horizon
        action_traj_comb = list(itertools.combinations_with_replacement(self.actions, horizon))
        action_traj_comb = random.sample(action_traj_comb, min(len(action_traj_comb), self.max_n_rand_traj))
        actions_values = np.ones(len(self.actions)) * float("inf")
        s = torch.FloatTensor(np.array([s]))
        for index_a, a in enumerate(self.actions):
            z_ = self.model.encode_next_state(s, torch.FloatTensor([[a]]))
            dist = self.min_dist_to_goal_states(z_, z_goal_states)
            for action_traj in action_traj_comb:
                assert len(action_traj) > 0
                dist = self.cont_action_step_lookahead(z_, action_traj, dist, z_goal_states)
                actions_values[index_a] = min(dist, actions_values[index_a])#(dist/(len(action_traj_comb)+1))

        return np.argmin(actions_values)

    def cont_prob_multi_step_lookahead(self, s, z_goal_states, temperature=1, verbose=False):
        horizon = self.max_horizon
        action_traj_comb = list(itertools.combinations_with_replacement(self.actions, horizon))
        action_traj_comb = random.sample(action_traj_comb, min(len(action_traj_comb), self.max_n_rand_traj))
        actions_values = np.ones(len(self.actions)) * float("inf")
        s = torch.FloatTensor(np.array([s]))
        for index_a, a in enumerate(self.actions):
            z_ = self.model.encode_next_state(s, torch.FloatTensor([a]))
            dist = self.min_dist_to_goal_states(z_, z_goal_states)
            for action_traj in action_traj_comb:
                assert len(action_traj) > 0
                dist = self.cont_action_step_lookahead(z_, action_traj, dist, z_goal_states)
                actions_values[index_a] = min(dist, actions_values[index_a])#(dist/(len(action_traj_comb)+1))
        m = torch.nn.Softmin(dim=0)
        softmin_p = m(torch.FloatTensor(actions_values/temperature)).detach().numpy()
        if verbose: print(softmin_p)
        r_a_i = np.random.choice(len(self.actions), p=softmin_p)
        return self.actions[r_a_i]
