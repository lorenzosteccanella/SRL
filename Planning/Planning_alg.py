import itertools
import random
import numpy as np
import torch

class Planner():

    def __init__(self, model: torch.nn.Module, actions: list, max_n_rand_traj: int = 10, max_horizon: int = 0):
        """
        A simple planner that unrolls the model for a given horizon and returns the action that minimizes the distance
        to the goal state. Only the first action is then taken. Similar to the MPC algorithm.
        Args:
            model: A model of the environment
            actions: A discrete list of actions
            max_n_rand_traj: The maximum number of random trajectories to sample
            max_horizon: The maximum horizon to unroll the model
        """
        self.model = model
        self.actions = actions
        self.max_n_rand_traj = max_n_rand_traj
        self.max_horizon = max_horizon

    def get_z_goal_states(self, goal_states: np.array) -> torch.FloatTensor:
        """
        To get the latent representation of the goal states
        Args:
            goal_states: A list of goal states

        Returns:
            The latent representation of the goal states
        """
        with torch.no_grad():
            z_goal_states = self.model.encode_state(torch.FloatTensor(goal_states))
        return z_goal_states

    def min_dist_to_goal_states(self, z: torch.Tensor, z_goal_states: torch.Tensor) -> float:
        """

        Args:
            z: The latent representation of the current state
            z_goal_states: The latent representation of the goal states

        Returns:
            The minimum distance between the current state and the goal states
        """
        with torch.no_grad():
            tmp_dist = float("inf")
            for z_goal_state in z_goal_states:
                z_dist = self.model.encoder.dist(z, z_goal_state).detach().numpy()[0]
                if tmp_dist > z_dist: tmp_dist = z_dist
            return tmp_dist

    def avg_dist_to_goal_states(self, z: torch.Tensor, z_goal_states: torch.Tensor) -> float:
        """
        The average distance between the current state and the goal states
        Args:
            z: The latent representation of the current state
            z_goal_states: The latent representation of the goal states

        Returns:
            The average distance between the current state and the goal states

        """
        with torch.no_grad():
            tmp_dist = 0
            for z_goal_state in z_goal_states:
                tmp_dist += self.model.encoder.dist(z_goal_state, z).detach().numpy()[0]
            return tmp_dist / len(z_goal_states)

    def one_hot_action(self, a: int) -> torch.Tensor:
        """
        The one hot encoding of an action
        Args:
            a: The action to encode
        Returns:
            The one hot encoding of the action
        """
        with torch.no_grad():
            one_hot_a = np.zeros(len(self.actions))
            one_hot_a[a] = 1
            one_hot_a = np.expand_dims(one_hot_a, axis=0)
            one_hot_a = torch.FloatTensor(one_hot_a)
            return one_hot_a

    def action_step_lookahead(self, z: torch.Tensor, action_traj: list, dist: int, z_goal_states: torch.Tensor) -> float:
        """
        To unroll the model for a given horizon and returns the sum of the distances to the goal states
        Args:
            z: The latent representation of the current state
            action_traj: The action trajectory to unroll the model
            dist: The distance to the goal states from the current state
            z_goal_states: The latent representation of the goal states

        Returns:
            The sum of the distances along the unrolled trajectory to the goal states

        """
        for a, step in zip(action_traj, range(self.max_horizon)):
            one_hot_a = self.one_hot_action(a)
            z_ = self.model.encode_action(z, one_hot_a)
            #dist += self.min_dist_to_goal_states(z_, z_goal_states)
            z = z_

        dist = self.min_dist_to_goal_states(z, z_goal_states)
        return dist

    def cont_action_step_lookahead(self, z: torch.Tensor, action_traj: list, dist: int, z_goal_states: torch.Tensor) -> float:
        """
        To unroll the model for a given horizon in case of continous or numerical actions and returns the
        sum of the distances to the goal states
        Args:
            z: The latent representation of the current state
            action_traj: The action trajectory to unroll the model
            dist: The distance to the goal states from the current state
            z_goal_states: The latent representation of the goal states

        Returns:
            The sum of the distances along the unrolled trajectory to the goal states

        """
        for a, step in zip(action_traj, range(self.max_horizon)):
            z_ = self.model.encode_action(z, torch.FloatTensor(np.array([a])))
            #dist += self.min_dist_to_goal_states(z_, z_goal_states)
            z = z_
        dist = self.min_dist_to_goal_states(z, z_goal_states)
        return dist

    def prob_multi_step_lookahead(self, s: list, z_goal_states: torch.Tensor, temperature: float = 0.1, verbose: bool =False) -> int:
        """
        To unroll the model for a given horizon and returns the action that minimizes the distance to the goal states
        Args:
            s: The current state
            z_goal_states: The latent representation of the goal states
            temperature: The parameter of the softmin function
            verbose: To print the probabilities of the actions

        Returns:
            The action that minimizes the distance to the goal states

        """
        with torch.no_grad():
            horizon = self.max_horizon
            action_traj_comb = list(itertools.combinations_with_replacement(self.actions, horizon))
            action_traj_comb = random.sample(action_traj_comb, min(len(action_traj_comb), self.max_n_rand_traj))
            actions_values = np.ones(len(self.actions)) * float("inf")
            s = torch.FloatTensor(np.array([s]))
            for a in self.actions:
                one_hot_a = self.one_hot_action(a)
                z_ = self.model.eval_encode_next_state(s, one_hot_a)
                dist = self.min_dist_to_goal_states(z_, z_goal_states)
                actions_values[a] = min(dist, actions_values[a])
                for action_traj in action_traj_comb:
                    if len(action_traj) > 0:
                        dist = self.action_step_lookahead(z_, action_traj, dist, z_goal_states)
                        actions_values[a] = min(dist, actions_values[a])  # (dist/(len(action_traj_comb)+1))
            m = torch.nn.Softmin(dim=0)
            softmin_p = m(torch.FloatTensor(actions_values / temperature)).detach().numpy()
            if verbose: print(softmin_p, actions_values)
            return np.random.choice(self.actions, p=softmin_p)

    def cont_prob_multi_step_lookahead(self, s: list, z_goal_states: torch.Tensor, temperature: float =0.1, verbose: bool =False) -> int:
        """
        Same as prob_multi_step_lookahead but for continous or numerical actions
        Args:
            s: The current state
            z_goal_states: The latent representation of the goal states
            temperature: The parameter of the softmin function
            verbose: To print the probabilities of the actions

        Returns:
            The action that minimizes the distance to the goal states

        """
        with torch.no_grad():
            horizon = self.max_horizon
            action_traj_comb = list(itertools.combinations_with_replacement(self.actions, horizon))
            action_traj_comb = random.sample(action_traj_comb, min(len(action_traj_comb), self.max_n_rand_traj))
            actions_values = np.ones(len(self.actions)) * float("inf")
            s = torch.FloatTensor(np.array([s]))
            for index_a, a in enumerate(self.actions):
                z_ = self.model.eval_encode_next_state(s, torch.FloatTensor(np.array([a])))
                dist = self.min_dist_to_goal_states(z_, z_goal_states)
                actions_values[index_a] = min(dist, actions_values[index_a])
                for action_traj in action_traj_comb:
                    if len(action_traj) > 0:
                        dist = self.cont_action_step_lookahead(z_, action_traj, dist, z_goal_states)
                        actions_values[index_a] = min(dist, actions_values[index_a])  # (dist/(len(action_traj_comb)+1))
            m = torch.nn.Softmin(dim=0)
            softmin_p = m(torch.FloatTensor(actions_values / temperature)).detach().numpy()
            if verbose: print(softmin_p)
            r_a_i = np.random.choice(len(self.actions), p=softmin_p)
            return self.actions[r_a_i]
