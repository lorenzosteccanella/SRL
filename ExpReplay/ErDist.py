import copy
import random
from collections import deque
import numpy as np
import torch
from tqdm import trange
from typing import Union, Tuple


class ErDist:
    """
    The buffer used to store trajectories and get batches to train the min action distance model
    """

    def __init__(self, max_n_trajectories: int = 10000, trajectories_list: list or None = None):
        """

        :param max_n_trajectories: The maximum size in number of trajectories we can store
        :param trajectories_list: Optional, A list of trajectories
        """
        self.trajectories = deque(maxlen=max_n_trajectories)
        if trajectories_list is not None: self.add_trajectories(trajectories_list)

    def add_trajectories(self, trajectories: list):
        """
        To add a list of trajectories to the buffer
        :param trajectories: The list of trajectories
        :return:
        """
        t = trange(len(trajectories), desc='Add trajectories to experience replay', leave=True)
        for n_traj in t:
            self.add_trajectory(trajectory=trajectories[n_traj])

    def add_trajectory(self, trajectory: list):
        """
        To add a single trajectory to the buffer
        :param trajectory: The trajectory to be added
        :return:
        """
        self.trajectories.append(copy.deepcopy(trajectory))

    def get_batch(self, batch_size: int, d_tresh: int or None = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        sample randomly a batch to train the distance
        :param d_tresh: a distance treshold
        :param batch_size: the batch size
        :return:
        """
        s1_b = []
        s2_b = []
        d_traj_b = []
        for i in range(batch_size):
            # select randomly a trajectory
            t = self.trajectories[np.random.randint(0, len(self.trajectories))]
            s1_i = np.random.randint(0, len(t))
            if d_tresh is None:
                s2_i = np.random.randint(0, len(t))
            else:
                s2_i = np.random.choice(list(range(s1_i, min((s1_i + d_tresh+1), (len(t)))))) #TODO verify this
            s1_i, s2_i = min(s1_i, s2_i), max(s1_i, s2_i)
            s1_b.append(t[s1_i]["s"])
            s2_b.append(t[s2_i]["s"])
            d_traj_b.append((s2_i - s1_i))

        return torch.FloatTensor(np.array(s1_b)), \
               torch.FloatTensor(np.array(s2_b)), \
               torch.FloatTensor(np.array(d_traj_b))

    def get_batch_action(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s1_b = []
        s2_b = []
        a_b = []
        for i in range(batch_size):
            t = self.trajectories[np.random.randint(0, len(self.trajectories))]
            s1_i = np.random.randint(0, len(t))
            s1_b.append(t[s1_i]["s"])
            s2_b.append(t[s1_i]["s_"])
            a_b.append(t[s1_i]["a"])

        return (torch.FloatTensor(np.array(s1_b)),
                torch.FloatTensor(np.array(s2_b)),
                torch.FloatTensor(np.array(a_b)))

