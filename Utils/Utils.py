import copy
import itertools
import math
import random
import time
import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt
from machin.frame.algorithms import DQNPer, DDPG
from machin.utils.logging import default_logger as logger
from machin.frame.buffers.prioritized_buffer import PrioritizedBuffer
from machin.frame.transition import TransitionBase
from torch.utils.data import Dataset
from pylab import figure
from mpl_toolkits.mplot3d import Axes3D

from Envs.PointmassEnv import PointmassEnv

plt.rcParams["figure.figsize"] = (20, 10)

def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n) and (n != "ld.weight"):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)

def collect_trajectories_GridWorld(env, n_of_trajectories=100, obs_noise=False):
    trajectories = []
    trajectory = []
    states = []

    for _ in range(n_of_trajectories):
        s = env.reset()
        if obs_noise:
            s = np.array(s) + np.abs(np.around(np.random.normal(loc=0.0, scale=0.1), 2))
        trajectory.append(s[0:-1])
        if tuple(s[0:-1]) not in states: states.append(tuple(s[0:-1]))

        while True:
            a = random.choice(list(range(env.action_space.n)))
            s_, _, done, _ = env.step(a)
            if obs_noise:
                s_ = np.array(s_) + np.abs(np.around(np.random.normal(loc=0.0, scale=0.1), 2))
            if tuple(s_[0:-1]) not in states: states.append(tuple(s_[0:-1]))
            #env.render()
            trajectory.append(s_[0:-1])
            if done: break

        trajectories.append(tuple(trajectory))
        trajectory.clear()
    return trajectories, states

def collect_trajectories(env, n_of_trajectories=100, obs_noise=False, max_n_steps=200):
    trajectories = []
    trajectory = []
    states = []

    for _ in range(n_of_trajectories):
        s = env.reset()
        if obs_noise:
            s = np.array(s) + np.abs(np.around(np.random.normal(loc=0.0, scale=0.1), 2))
        trajectory.append(s)
        if tuple(s) not in states: states.append(tuple(s))
        steps = 0
        while steps < max_n_steps:
            a = random.choice(list(range(env.action_space.n)))
            s_, _, done, _ = env.step(a); steps += 1
            if obs_noise:
                s_ = np.array(s_) + np.abs(np.around(np.random.normal(loc=0.0, scale=0.1), 2))
            if tuple(s_) not in states: states.append(tuple(s_))
            #env.render()
            trajectory.append(s_)
            if done: break

        trajectories.append(tuple(trajectory))
        trajectory.clear()
    return trajectories, states

def collect_pos_trajectories(env, n_of_trajectories=100, obs_noise=False):
    trajectories = []
    trajectory = []
    states = []

    for _ in range(n_of_trajectories):
        s = env.reset()["pos"]
        if obs_noise:
            s = np.array(s) + np.abs(np.around(np.random.normal(loc=0.0, scale=0.1), 2))
        trajectory.append(s)
        if tuple(s) not in states: states.append(tuple(s))

        while True:
            a = random.choice(list(range(env.action_space.n)))
            s_, _, done, _ = env.step(a)
            s_ = s_["pos"]
            if obs_noise:
                s_ = np.array(s_) + np.abs(np.around(np.random.normal(loc=0.0, scale=0.1), 2))
            if tuple(s_) not in states: states.append(tuple(s_))
            #env.render()
            trajectory.append(s_)
            if done: break

        trajectories.append(tuple(trajectory))
        trajectory.clear()
    return trajectories, states

def collect_action_trajectories(env, n_of_trajectories=100, obs_noise=False, max_n_steps=200):
    trajectories = []
    trajectory = []
    states = []

    for _ in range(n_of_trajectories):
        s = env.reset()
        if tuple(s) not in states: states.append(tuple(s))
        steps = 0
        while steps < max_n_steps:
            a = random.choice(list(range(env.action_space.n)))
            one_hot_a = np.zeros(env.action_space.n)
            one_hot_a[a] = 1
            s_, _, done, _ = env.step(a); steps += 1
            if tuple(s_) not in states: states.append(tuple(s_))
            trajectory.append(tuple((s, copy.deepcopy(one_hot_a))))
            s = s_
            if done: break
        one_hot_a = np.zeros(env.action_space.n)
        trajectory.append(tuple((s_, copy.deepcopy(one_hot_a))))  # DO I NEED THIS OR NOT??? THIS WOULD BE A TERMINAL STATE!!!
        trajectories.append(tuple(trajectory))
        trajectory.clear()
    return trajectories, states

def collect_continuous_action_trajectories(env, n_of_trajectories=100, r_action_fn=None):
    trajectories = []
    trajectory = []

    for i in range(n_of_trajectories):
        s, _ = env.reset()
        s= s["observation"][:3]
        steps = 0
        done = False
        while not done:
            # sample a numpy 4 dimensional vector
            a = np.random.uniform(-0.5, 1, 4)
            #a = env.action_space.sample()
            a[3] = 0
            out = env.step(a); steps += 1
            s_, _, _, done, _ = out
            s_ = s_["observation"][:3]
            trajectory.append(tuple((s, copy.deepcopy(a))))
            s = s_
        trajectory.append(tuple((s_, np.array([0,0,0,0]))))  # DO I NEED THIS OR NOT??? THIS WOULD BE A TERMINAL STATE!!!
        print(len(trajectory))
        trajectories.append(tuple(trajectory))
        trajectory.clear()
    return trajectories

def collect_pos_action_trajectories(env, n_of_trajectories=100, obs_noise=False):
    trajectories = []
    trajectory = []
    states = []

    for _ in range(n_of_trajectories):
        s = env.reset()["pos"]
        if obs_noise:
            mean = [0, 0, 0, 0]
            cov = [[0.1, 0, 0, 0], [0, 0.1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
            s = np.array(s) + np.random.multivariate_normal(mean, cov, 1)[0]
        if tuple(s) not in states: states.append(tuple(s))

        while True:
            a = random.choice(list(range(env.action_space.n)))
            one_hot_a = np.zeros(env.action_space.n)
            one_hot_a[a] = 1
            s_, _, done, _ = env.step(a)
            s_ = s_["pos"]
            if obs_noise:
                mean = [0, 0, 0, 0]
                cov = [[0.1, 0, 0, 0], [0, 0.1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
                s_ = np.array(s_) + np.random.multivariate_normal(mean, cov, 1)[0]
            if tuple(s_) not in states: states.append(tuple(s_))
            # env.render()
            # print(s, a, s_)
            # time.sleep(1)
            trajectory.append(tuple((s, copy.deepcopy(one_hot_a))))
            s = s_
            if done: break
        one_hot_a = np.zeros(env.action_space.n)
        trajectory.append(tuple((s_, copy.deepcopy(one_hot_a))))  # DO I NEED THIS OR NOT??? THIS WOULD BE A TERMINAL STATE!!!
        trajectories.append(tuple(trajectory))
        trajectory.clear()
    return trajectories, states

def collect_image_trajectories(env, n_of_trajectories=100, obs_noise=None):
    trajectories = []
    trajectory = []
    states = []
    pos_unique_states = []

    for _ in range(n_of_trajectories):
        s = env.reset()
        trajectory.append(s["image"])
        if tuple(s["pos"]) not in pos_unique_states:
            pos_unique_states.append(tuple(s["pos"]))
            states.append(s["image"])

        while True:
            a = random.choice(list(range(env.action_space.n)))
            s_, _, done, _ = env.step(a)
            if tuple(s_["pos"]) not in pos_unique_states:
                pos_unique_states.append(tuple(s["pos"]))
                states.append(s_["image"])
            #env.render()
            trajectory.append(s_["image"])
            if done: break

        trajectories.append(tuple(trajectory))
        trajectory.clear()
    return trajectories, states

def save_trajectories(trajectories):
    trajectories = np.array(trajectories, dtype=object)
    np.save("../trajectories.npy", trajectories)

def load_trajectories(path=None):
    if path is None:
        trajectories = np.load("../trajectories.npy", allow_pickle=True)
        return trajectories
    else:
        trajectories = np.load(path, allow_pickle=True)
        return trajectories

def save_action_trajectories(trajectories, path= None):
    if path is None:
        trajectories = np.array(trajectories, dtype=object)
        np.save("../action_trajectories.npy", trajectories)
    else:
        trajectories = np.array(trajectories, dtype=object)
        np.save(path, trajectories)

def load_action_trajectories(path=None):
    if path is None:
        trajectories = np.load("../action_trajectories.npy", allow_pickle=True)
        return trajectories
    else:
        trajectories = np.load(path, allow_pickle=True)
        return trajectories

def plot_result2(z, annotation, d, split_coeficent=0.01, show_annotations=True):

    plot_x = []
    plot_y = []
    if d == 1 or d == 2:
        fig, ax = plt.subplots()
    if d == 3:
        fig = plt.figure()
        ax = Axes3D(fig)
    unique_states = []
    unique_positions = {}
    for state, txt in zip(z, annotation):
        state = tuple(state)
        txt = txt
        if state not in unique_states:

            if d == 3:
                ax.scatter(state[0], state[1], state[2], color='b')
            if d == 2:
                plot_x.append(state[0])
                plot_y.append(state[1])
            if d == 1:
                plot_x.append(state[0])
                plot_y.append(0)

            unique_states.append(state)

        if d == 3:
            if (round(state[0], 1), round(state[1], 1), round(state[2], 1)) not in unique_positions.keys():
                unique_positions[(round(state[0], 1), round(state[1], 1), round(state[2], 1))] = 0
            if show_annotations: ax.text(state[0], state[1], state[2], txt, size=20, zorder=1, color='k')
            unique_positions[(round(state[0], 1), round(state[1], 1), round(state[2], 1))] += 1
        if d == 2:
            if (round(state[0], 1), round(state[1], 1)) not in unique_positions.keys():
                unique_positions[(round(state[0], 1), round(state[1], 1))] = 0
            split_value = unique_positions[(round(state[0], 1), round(state[1], 1))]
            if show_annotations: ax.annotate(txt, (state[0], state[1] + split_coeficent * split_value))
            unique_positions[(round(state[0], 1), round(state[1], 1))] += 1
        if d == 1:
            if (round(state[0], 1), 0) not in unique_positions.keys():
                unique_positions[(round(state[0], 1), 0)] = 0
            split_value = unique_positions[(round(state[0], 1), 0)]
            if show_annotations: ax.annotate(txt, (state[0], 0 + split_coeficent * split_value))
            unique_positions[(round(state[0], 1), 0)]+= 1

    if d==1 or d==2:
        ax.scatter(plot_x, plot_y)

    plt.show()

def plot_result3(z, annotation, d, split_coeficent=0.01, show_annotations=True):

    plot_x = []
    plot_y = []
    plot_z = []
    if d == 1 or d == 2:
        fig, ax = plt.subplots()
    if d == 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
    unique_states = []
    unique_positions = {}
    for state, txt in zip(z, annotation):
        state = tuple(state)
        txt = txt
        if state not in unique_states:
            if d == 3:
                plot_x.append(state[0])
                plot_y.append(state[1])
                plot_z.append(state[2])
            if d == 2:
                plot_x.append(state[0])
                plot_y.append(state[1])
            if d == 1:
                plot_x.append(state[0])
                plot_y.append(0)

            unique_states.append(state)

        if d == 3:
            if (round(state[0], 1), round(state[1], 1), round(state[2], 1)) not in unique_positions.keys():
                unique_positions[(round(state[0], 1), round(state[1], 1), round(state[2], 1))] = 0
            unique_positions[(round(state[0], 1), round(state[1], 1), round(state[2], 1))] += 1
        if d == 2:
            if (round(state[0], 1), round(state[1], 1)) not in unique_positions.keys():
                unique_positions[(round(state[0], 1), round(state[1], 1))] = 0
            split_value = unique_positions[(round(state[0], 1), round(state[1], 1))]
            if show_annotations: ax.annotate(txt, (state[0], state[1] + split_coeficent * split_value))
            unique_positions[(round(state[0], 1), round(state[1], 1))] += 1
        if d == 1:
            if (round(state[0], 1), 0) not in unique_positions.keys():
                unique_positions[(round(state[0], 1), 0)] = 0
            split_value = unique_positions[(round(state[0], 1), 0)]
            if show_annotations: ax.annotate(txt, (state[0], 0 + split_coeficent * split_value))
            unique_positions[(round(state[0], 1), 0)]+= 1

    if d==1 or d==2:
        ax.scatter(plot_x, plot_y, s=200)
    if d==3:
        ax.scatter(plot_x, plot_y, plot_z)
    plt.show()

def get_plot_result3(z, annotation, d, split_coeficent=0.01, show_annotations=True):

    plot_x = []
    plot_y = []
    plot_z = []
    if d == 1 or d == 2:
        fig, ax = plt.subplots()
    if d == 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
    unique_states = []
    unique_positions = {}
    for state, txt in zip(z, annotation):
        state = tuple(state)
        txt = txt
        if state not in unique_states:
            if d == 3:
                plot_x.append(state[0])
                plot_y.append(state[1])
                plot_z.append(state[2])
            if d == 2:
                plot_x.append(state[0])
                plot_y.append(state[1])
            if d == 1:
                plot_x.append(state[0])
                plot_y.append(0)

            unique_states.append(state)

        if d == 3:
            if (round(state[0], 1), round(state[1], 1), round(state[2], 1)) not in unique_positions.keys():
                unique_positions[(round(state[0], 1), round(state[1], 1), round(state[2], 1))] = 0
            unique_positions[(round(state[0], 1), round(state[1], 1), round(state[2], 1))] += 1
        if d == 2:
            if (round(state[0], 1), round(state[1], 1)) not in unique_positions.keys():
                unique_positions[(round(state[0], 1), round(state[1], 1))] = 0
            split_value = unique_positions[(round(state[0], 1), round(state[1], 1))]
            if show_annotations: ax.annotate(txt, (state[0], state[1] + split_coeficent * split_value))
            unique_positions[(round(state[0], 1), round(state[1], 1))] += 1
        if d == 1:
            if (round(state[0], 1), 0) not in unique_positions.keys():
                unique_positions[(round(state[0], 1), 0)] = 0
            split_value = unique_positions[(round(state[0], 1), 0)]
            if show_annotations: ax.annotate(txt, (state[0], 0 + split_coeficent * split_value))
            unique_positions[(round(state[0], 1), 0)]+= 1

    if d==1 or d==2:
        ax.scatter(plot_x, plot_y)
        return fig
    if d==3:
        ax.scatter(plot_x, plot_y, plot_z)
        return fig
    else:
        return None

def plot_to_wandb(fig, name):
    """
    Takes a matplotlib figure handle and converts it using
    canvas and string-casts to a numpy array that can be
    visualized in TensorBoard using the add_image function

    Parameters:
        wandb_writer (tensorboard.SummaryWriter): wandb instance.
        fig (matplotlib.pyplot.fig): Matplotlib figure handle.
        step (int): counter usually specifying steps/epochs/time.
    """

    # Draw figure on canvas
    fig.canvas.draw()

    # Convert the figure to numpy array, read the pixel values and reshape the array
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # Add figure in numpy "image" to TensorBoard writer
    wandb.log({name: [wandb.Image(img)]})
    plt.close(fig)

def plot_result(z, annotation, d, split_coeficent=0.01, show_annotations=True):

    plot_x = []
    plot_y = []
    plot_z = []
    if d == 1 or d == 2:
        fig, ax = plt.subplots()
    if d == 3:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
    unique_states = []
    unique_annotations = []
    unique_positions = {}
    for state, txt in zip(z, annotation):
        state = tuple(state)
        txt = txt
        if state not in unique_states:

            if d == 3:
                plot_x.append(state[0])
                plot_y.append(state[1])
                plot_z.append(state[2])
            if d == 2:
                plot_x.append(state[0])
                plot_y.append(state[1])
            if d == 1:
                plot_x.append(state[0])
                plot_y.append(0)

            unique_states.append(state)

        if txt not in unique_annotations:

            if d == 3:
                if (round(state[0], 1), round(state[1], 1), round(state[2], 1)) not in unique_positions.keys():
                    unique_positions[(round(state[0], 1), round(state[1], 1), round(state[2], 1))] = 0
                unique_positions[(round(state[0], 1), round(state[1], 1), round(state[2], 1))] += 1
            if d == 2:
                if (round(state[0], 1), round(state[1], 1)) not in unique_positions.keys():
                    unique_positions[(round(state[0], 1), round(state[1], 1))] = 0
                split_value = unique_positions[(round(state[0], 1), round(state[1], 1))]
                if show_annotations: ax.annotate(txt, (state[0], state[1] + split_coeficent * split_value))
                unique_positions[(round(state[0], 1), round(state[1], 1))] += 1
            if d == 1:
                if (round(state[0], 1), 0) not in unique_positions.keys():
                    unique_positions[(round(state[0], 1), 0)] = 0
                split_value = unique_positions[(round(state[0], 1), 0)]
                if show_annotations: ax.annotate(txt, (state[0], 0 + split_coeficent * split_value))
                unique_positions[(round(state[0], 1), 0)]+= 1

            unique_annotations.append(txt)

    if d==1 or d==2:
        ax.scatter(plot_x, plot_y)
    if d==3:
        ax.scatter(plot_x, plot_y, plot_z)
    plt.show()

def quad_feature_transf(x, prev_transf_template = None):
    new_x = []
    template = []
    if prev_transf_template:
        template = copy.deepcopy(prev_transf_template)
    for i, state in enumerate(x):
        new_state = []
        for k, x_k in enumerate(state):
            new_state.append(x_k)
            if i == 0: template.append("x"+str(k))
        for k, x_k in enumerate(state):
            new_state.append(x_k**2)
            if i == 0: template.append("x"+str(k)+"**2")
            for q, x_q in enumerate(state[k+1::]):
                new_state.append(x_k * x_q)
                if i == 0: template.append("x"+str(k)+"x"+str(q))
        new_x.append(np.array(new_state))

    return np.array(new_x), template

def neg_transf(x):
    new_x = []
    for i, state in enumerate(x):
        new_state = []
        for k, x_k in enumerate(state):
            new_state.append(x_k)
        for k, x_k in enumerate(state):
            new_state.append(-x_k)

        new_x.append(np.array(new_state))

    return np.array(new_x)

def scal_transf(x, max_scal=10, prev_transf_template = None):
    new_x = []
    template = []
    if prev_transf_template:
        template = copy.deepcopy(prev_transf_template)
    for i, state in enumerate(x):
        new_state=[]
        for k, x_k in enumerate(state):
            new_state.append(x_k)
        for k, x_k in enumerate(state):
            for j in range(2, max_scal+1):
                new_state.append(x_k * j)
                new_state.append(x_k * -j)
                if i == 0 and prev_transf_template: template.append(prev_transf_template[k]+" * "+str(j))
                if i == 0 and prev_transf_template: template.append(prev_transf_template[k]+" * "+str(-j))

        new_x.append(np.array(new_state))

    return np.array(new_x), template

def add_bias(x):
    new_x = []
    for i, state in enumerate(x):
        new_state = [1, ]
        for x_i in x:
            new_state.append(x_i)

        new_x.append(np.array(new_state))

    return np.array(new_x)

class PER():
    def __init__(self, trajectories, replay_size=1000000, len_window_o=None, c_window_len=1, d_sampler=1, percent_random_sample=1, ub_w_dist=10):
        self.dataset = PrioritizedBuffer(replay_size, "cpu", epsilon=0.1, alpha=0.6, beta=1, beta_increment_per_sampling=0)
        for n_traj, trajectory in enumerate(trajectories):
            print("loading trajectory: ", n_traj)
            tmp_observations = []
            indexes_comb = list(itertools.combinations(list(range(len(trajectory))), r=2))
            if len_window_o is not None: indexes_comb = [i for i in indexes_comb if (i[1] - i[0]) <= len_window_o]
            if percent_random_sample is not None:
                indexes_comb = random.sample(indexes_comb, int(len(indexes_comb)*percent_random_sample))
            for i in indexes_comb:

                d_o = min((i[1] - i[0]), ub_w_dist)
                o = (trajectory[i[0]], trajectory[i[1]], d_o)

                if i[0]+1 < len(trajectory):
                    c_f = (trajectory[i[0]], trajectory[i[0]+1])
                else:
                    c_f = (trajectory[0], trajectory[1])  # just to ensuere that c_f this is never empty

                d_c = (i[1] - i[0])
                c = (trajectory[i[0]], trajectory[i[1]], d_c)

                data = {"x1_o": (o[0][0]),
                        "x2_o": (o[1][0]),
                        "x1_a_o": (o[0][1]),
                        "x2_a_o": (o[1][1]),
                        "d_obj": (o[2]),
                        "x1_c": (c[0][0]),
                        "x2_c": (c[1][0]),
                        "x1_a_c": (c[0][1]),
                        "x2_a_c": (c[1][1]),
                        "d_c": (c[2]),
                        "x1_f": (c_f[0][0]),
                        "x2_f": (c_f[1][0]),
                        "x1_a_f": (c_f[0][1]),
                        "x2_a_f": (c_f[1][1])}
                tmp_observations.append(TransitionBase([], [], data.keys(), [], [], data.values()))

            self.dataset.store_episode(tmp_observations, required_attrs=("x1_o", "x2_o", "x1_a_o", "x2_a_o", "d_obj",
                                                                         "x1_c", "x2_c", "x1_a_c", "x2_a_c", "d_c",
                                                                         "x1_f", "x2_f", "x1_a_f", "x2_a_f"),
                                       priorities=list(np.ones(len(tmp_observations))))

class dataset_w_c(Dataset):

    def __init__(self, trajectories, len_window_o=None, c_window_len=1, d_sampler=1, percent_random_sample=1, ub_w_dist=10):
        self.dataset = []
        #all_traj_ind = list(range(len(trajectories)))
        for n_traj, trajectory in enumerate(trajectories):
            print("loading trajectory: ", n_traj)
            indexes_comb = list(itertools.combinations(list(range(len(trajectory))), r=2))
            if len_window_o is not None: indexes_comb = [i for i in indexes_comb if (i[1] - i[0]) <= len_window_o]
            if percent_random_sample is not None:
                # p = np.array([(i[1] - i[0]) for i in indexes_comb])
                # p = p/p.sum()
                # idx_comb = np.random.choice(np.array(list(range(len(indexes_comb)))), size=int(len(indexes_comb)*percent_random_sample), replace=False, p=p)
                # indexes_comb = np.array(indexes_comb)[idx_comb, :]
                indexes_comb = random.sample(indexes_comb, int(len(indexes_comb)*percent_random_sample))
            for i in indexes_comb:

                d_o = min((i[1] - i[0]), ub_w_dist)
                o = (trajectory[i[0]], trajectory[i[1]], d_o)

                # if len_window_o is None:
                #     d_o = 1 #max(1, (i[1] - i[0]) // d_sampler)
                #     o = (trajectory[i[0]], trajectory[i[1]], d_o)
                # elif (i[1] - i[0]) <= len_window_o:
                #     d_o = 1 #max(1, (i[1] - i[0]) // d_sampler)
                #     o = (trajectory[i[0]], trajectory[i[1]], d_o)

                # traj_ind = random.sample(all_traj_ind, int(len(trajectories)*percent_random_sample))
                # o_r = []
                # for t_i in traj_ind:
                #     i_s = random.sample(list(range(len(trajectories[t_i]))), 1)[0]
                #     o_r.append((trajectory[i[0]], trajectories[t_i][i_s]))
                if i[0]+1 < len(trajectory):
                    c_f = (trajectory[i[0]], trajectory[i[0]+1])
                else:
                    c_f = (trajectory[0], trajectory[1])  # just to ensuere that c_f this is never empty

                # c=[]
                # for c_i in range(1, c_window_len+1):
                #     if i[0] - c_i >= 0:
                #         d_c = max(0, c_i // d_sampler)
                #         c.append((trajectory[i[0]], trajectory[i[0] - c_i], d_c))
                #     if i[1] - c_i >= 0:
                #         d_c = max(0, c_i // d_sampler)
                #         c.append((trajectory[i[1]], trajectory[i[1] - c_i], d_c))
                #     if i[0] + c_i < len(trajectory):
                #         d_c = max(0, c_i // d_sampler)
                #         c.append((trajectory[i[0]], trajectory[i[0] + c_i], d_c))
                #     if i[1] + c_i < len(trajectory):
                #         d_c = max(0, c_i // d_sampler)
                #         c.append((trajectory[i[1]], trajectory[i[1] + c_i], d_c))
                #     # else:
                #     #     d_c = max(0, c_i // d_sampler)
                #     #     c.append((trajectory[i[0]], trajectory[i[0]], d_c))

                # alternative version
                c = []
                d_c = (i[1] - i[0])
                c.append((trajectory[i[0]], trajectory[i[1]], d_c))
                for c_i in range(1, c_window_len+1):
                    if i[0] - c_i >= 0:
                        d_c = max(0, c_i // d_sampler)
                        c.append((trajectory[i[0]], trajectory[i[0] - c_i], d_c))
                    if i[1] - c_i >= 0:
                        d_c = max(0, c_i // d_sampler)
                        c.append((trajectory[i[1]], trajectory[i[1] - c_i], d_c))
                    if i[0] + c_i < len(trajectory):
                        d_c = max(0, c_i // d_sampler)
                        c.append((trajectory[i[0]], trajectory[i[0] + c_i], d_c))
                    if i[1] + c_i < len(trajectory):
                        d_c = max(0, c_i // d_sampler)
                        c.append((trajectory[i[1]], trajectory[i[1] + c_i], d_c))
                self.dataset.append((o, np.array(c), np.array(c_f)))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        o, c, c_f = self.dataset[idx]
        o1 = o[0]
        o2 = o[1]
        d_o = o[2]
        c_1 = []
        c_2 = []
        d_c = []
        r1 = []
        r2 = []
        c_f_1 = c_f[0]
        c_f_2 = c_f[1]
        for elem in c:
            c_1.append(elem[0])
            c_2.append(elem[1])
            d_c.append(elem[2])

        return o1, o2, d_o, c_1, c_2, d_c, c_f_1, c_f_2

def my_collate(batch):
    x1_o = []
    x2_o = []
    d_obj = []
    x1_c = []
    x2_c = []
    d_c = []
    x1_r = []
    x2_r = []
    for elem in batch:
        x1_o.append(elem[0])
        x2_o.append(elem[1])
        d_obj.append(elem[2])
        for e in elem[3]:
            x1_c.append(e)
        for e in elem[4]:
            x2_c.append(e)
        for e in elem[5]:
            d_c.append(e)
        for e in elem[6]:
            x1_r.append(e)
        for e in elem[7]:
            x2_r.append(e)

    return x1_o, x2_o, torch.FloatTensor(d_obj), x1_c, x2_c, torch.FloatTensor(d_c), x1_r, x2_r

def my_collate_v2(batch):
    x1_o = []
    x2_o = []
    x1_a_o = []
    x2_a_o = []
    d_obj = []
    x1_c = []
    x2_c = []
    x1_a_c = []
    x2_a_c = []
    d_c = []
    x1_f = []
    x2_f = []
    x1_a_f = []
    x2_a_f = []
    for elem in batch:
        x1_o.append(elem[0][0])
        x2_o.append(elem[1][0])
        x1_a_o.append(elem[0][1])
        x2_a_o.append(elem[1][1])
        d_obj.append(elem[2])
        for de, e1, e2 in zip(elem[5], elem[3], elem[4]):
            x1_c.append(e1[0])
            x1_a_c.append(e1[1])
            x2_c.append(e2[0])
            x2_a_c.append(e2[1])
            d_c.append(de)
        x1_f.append(elem[6][0])
        x2_f.append(elem[7][0])
        x1_a_f.append(elem[6][1])
        x2_a_f.append(np.zeros_like(elem[7][1]))

    return x1_o, x1_a_o, x2_o, x2_a_o, torch.FloatTensor(d_obj), x1_c, x1_a_c, x2_c, x2_a_c, torch.FloatTensor(d_c), x1_f, x1_a_f, x2_f, x2_a_f

def dqn_mountain_car_collect_trajectories(env, n_of_trajectories=40, obs_noise=None, action_state = False, discrete = False, scale_factor=1.0):
    # configurations
    observe_dim = env.observation_space.shape[0]
    action_num = env.action_space.n
    max_episodes = 1000
    max_steps = 200

    dataset_trajectories = dict()
    trajectory = []
    states = []
    trajectories = []

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

    q_net = QNet(observe_dim, action_num)
    q_net_t = QNet(observe_dim, action_num)

    dqn = DQNPer(q_net, q_net_t, torch.optim.Adam, torch.nn.MSELoss(reduction="sum"))

    episode, step, reward_fulfilled = 0, 0, 0
    smoothed_total_reward = -200

    list_ep_reward = []

    while True:
        episode += 1
        total_reward = 0
        terminal = False
        step = 0
        state = env.reset(); s=state * scale_factor
        #trajectory.append(state)
        if tuple(state) not in states: states.append(tuple(state))
        state = torch.tensor(state, dtype=torch.float32).view(1, observe_dim)
        episode_transitions = []

        while not terminal and step <= max_steps:
            step += 1
            with torch.no_grad():
                old_state = state
                # agent model inference
                action = dqn.act_discrete_with_noise({"some_state": old_state})
                state, reward, terminal, _ = env.step(action.item()); s_ = state * scale_factor;
                fake_reward = float(100 * (
                            (math.sin(3 * state[0]) * 0.0025 + 0.5 * state[1] * state[1]) - (
                                math.sin(3 * old_state[0, 0]) * 0.0025 + 0.5 * old_state[0, 1] * old_state[0, 1])).detach().numpy())
                #trajectory.append(state)
                if tuple(state) not in states: states.append(tuple(state))
                state = torch.tensor(state, dtype=torch.float32).view(1, observe_dim)
                total_reward += reward
                #env.render()

                if not discrete:
                    if action_state:
                        one_hot_a = np.zeros(env.action_space.n)
                        one_hot_a[action] = 1

                        trajectory.append(tuple((s, one_hot_a)))
                    else:
                        trajectory.append(s)

                if discrete:
                    if action_state:
                        one_hot_a = np.zeros(env.action_space.n)
                        one_hot_a[action] = 1
                        s = tuple(np.round((s - env.observation_space.low) * np.array([10, 100]), 0).astype(int))
                        trajectory.append(tuple((s, one_hot_a)))
                    else:
                        s = tuple(np.round((s - env.observation_space.low) * np.array([10, 100]), 0).astype(int))
                        trajectory.append(s)

                s = s_
                episode_transitions.append(
                    {
                        "state": {"some_state": old_state},
                        "action": {"action": action},
                        "next_state": {"some_state": state},
                        "reward": fake_reward,
                        "terminal": terminal or step == max_steps,
                    }
                )
        dqn.store_episode(episode_transitions)
        if not discrete:
            if action_state:
                one_hot_a = np.zeros(env.action_space.n)
                trajectory.append(tuple((s_, one_hot_a)))
                pass
            else:
                trajectory.append(s_)
        if discrete:
            if action_state:
                s_ = tuple(np.round((s_ - env.observation_space.low) * np.array([10, 100]), 0).astype(int))
                one_hot_a = np.zeros(env.action_space.n)
                trajectory.append(tuple((s_, one_hot_a)))
                pass
            else:
                s_ = tuple(np.round((s_ - env.observation_space.low) * np.array([10, 100]), 0).astype(int))
                trajectory.append(s_)



        # update, update more if episode is longer, else less
        if episode > 40:
            for _ in range(step):
                dqn.update()

        # show reward
        smoothed_total_reward = smoothed_total_reward * 0.9 + total_reward * 0.1
        logger.info(f"Episode {episode} total reward={smoothed_total_reward:.2f}")

        if smoothed_total_reward//1 not in dataset_trajectories.keys():
            print("\n\n", smoothed_total_reward//1, len(dataset_trajectories))
            dataset_trajectories[smoothed_total_reward//1] = episode_transitions
            #trajectories.append(tuple(trajectory))

        if episode > 40:
            list_ep_reward.append(total_reward)
            print("reward:", sum(list_ep_reward) / (episode-40))
            print("max_reward:", max(list_ep_reward))
            trajectories.append(tuple(trajectory))

        trajectory.clear()

        if len(trajectories) == n_of_trajectories: break

    return trajectories, states

def dqn_cart_pole_collect_trajectories(env, n_of_trajectories=40, obs_noise=None, action_state = False, discrete = False, scale_factor=1.0):
    # configurations
    observe_dim = env.observation_space.shape[0]
    action_num = env.action_space.n
    max_episodes = 1000
    max_steps = 200

    dataset_trajectories = dict()
    trajectory = []
    states = []
    trajectories = []

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

    q_net = QNet(observe_dim, action_num)
    q_net_t = QNet(observe_dim, action_num)

    dqn = DQNPer(q_net, q_net_t, torch.optim.Adam, torch.nn.MSELoss(reduction="sum"))

    episode, step, reward_fulfilled = 0, 0, 0
    smoothed_total_reward = 0

    list_ep_reward=[]

    while True:
        episode += 1
        total_reward = 0
        terminal = False
        step = 0
        state = env.reset(); s=state * scale_factor
        #trajectory.append(state)
        if tuple(state) not in states: states.append(tuple(state))
        state = torch.tensor(state, dtype=torch.float32).view(1, observe_dim)
        episode_transitions = []

        while not terminal and step <= max_steps:
            step += 1
            with torch.no_grad():
                old_state = state
                # agent model inference
                action = dqn.act_discrete_with_noise({"some_state": old_state})
                state, reward, terminal, _ = env.step(action.item()); s_ = state * scale_factor
                if tuple(state) not in states: states.append(tuple(state))
                state = torch.tensor(state, dtype=torch.float32).view(1, observe_dim)
                total_reward += reward
                env.render()

                if action_state:
                    one_hot_a = np.zeros(env.action_space.n)
                    one_hot_a[action] = 1

                    trajectory.append(tuple((s, one_hot_a)))
                else:
                    trajectory.append(s)

                s = s_
                episode_transitions.append(
                    {
                        "state": {"some_state": old_state},
                        "action": {"action": action},
                        "next_state": {"some_state": state},
                        "reward": reward,
                        "terminal": terminal or step == max_steps,
                    }
                )
        dqn.store_episode(episode_transitions)
        if action_state:
            one_hot_a = np.zeros(env.action_space.n)
            trajectory.append(tuple((s_, one_hot_a)))
            pass
        else:
            trajectory.append(s_)

        # update, update more if episode is longer, else less
        if episode > 40:
            for _ in range(step):
                dqn.update()

        # show reward
        smoothed_total_reward = smoothed_total_reward * 0.9 + total_reward * 0.1
        logger.info(f"Episode {episode} total reward={smoothed_total_reward:.2f}")
        if episode > 40:
            trajectories.append(tuple(trajectory))
            list_ep_reward.append(total_reward)
            print("reward:", sum(list_ep_reward) / len(list_ep_reward))
            print("max_reward:", max(list_ep_reward))

        trajectory.clear()

        if len(trajectories) == n_of_trajectories: break

    return trajectories, states

def dqn_acrobot_collect_trajectories(env, n_of_trajectories=40, obs_noise=None, action_state = False, discrete = False, scale_factor=1.0):
    # configurations
    observe_dim = env.observation_space.shape[0]
    action_num = env.action_space.n
    max_episodes = 1000
    max_steps = 200

    dataset_trajectories = dict()
    trajectory = []
    states = []
    trajectories = []

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

    q_net = QNet(observe_dim, action_num)
    q_net_t = QNet(observe_dim, action_num)

    dqn = DQNPer(q_net, q_net_t, torch.optim.Adam, torch.nn.MSELoss(reduction="sum"))

    episode, step, reward_fulfilled = 0, 0, 0
    smoothed_total_reward = 0

    Avg_tot_reward = 0
    tot_steps = 0

    while True:
        episode += 1
        total_reward = 0
        terminal = False
        step = 0
        state = env.reset(); s=state * scale_factor
        #trajectory.append(state)
        if tuple(state) not in states: states.append(tuple(state))
        state = torch.tensor(state, dtype=torch.float32).view(1, observe_dim)
        episode_transitions = []

        while not terminal and step <= max_steps:
            step += 1
            with torch.no_grad():
                old_state = state
                # agent model inference
                action = dqn.act_discrete_with_noise({"some_state": old_state})
                state, reward, terminal, _ = env.step(action.item()); s_ = state * scale_factor; tot_steps += 1; Avg_tot_reward += reward
                print(tot_steps/Avg_tot_reward)
                if tuple(state) not in states: states.append(tuple(state))
                state = torch.tensor(state, dtype=torch.float32).view(1, observe_dim)
                total_reward += reward
                #env.render()

                if action_state:
                    one_hot_a = np.zeros(env.action_space.n)
                    one_hot_a[action] = 1

                    trajectory.append(tuple((s, one_hot_a)))
                else:
                    trajectory.append(s)

                s = s_
                episode_transitions.append(
                    {
                        "state": {"some_state": old_state},
                        "action": {"action": action},
                        "next_state": {"some_state": state},
                        "reward": reward,
                        "terminal": terminal or step == max_steps,
                    }
                )
        dqn.store_episode(episode_transitions)
        if action_state:
            one_hot_a = np.zeros(env.action_space.n)
            trajectory.append(tuple((s_, one_hot_a)))
            pass
        else:
            trajectory.append(s_)

        # update, update more if episode is longer, else less
        if episode > 40:
            for _ in range(step):
                dqn.update()

        # show reward
        smoothed_total_reward = smoothed_total_reward * 0.9 + total_reward * 0.1
        logger.info(f"Episode {episode} total reward={smoothed_total_reward:.2f}")
        if episode > 200:
            trajectories.append(tuple(trajectory))

        trajectory.clear()

        if len(trajectories) == n_of_trajectories: break

    return trajectories, states

def ddpg_pendulum_collect_trajectories(env, n_of_trajectories=40, obs_noise=None, action_state = False, discrete = False, scale_factor=1.0):
    # configurations
    observe_dim = 3
    action_dim = 1
    action_range = 2
    max_steps = 200
    noise_param = (0, 0.2)
    noise_mode = "normal"

    dataset_trajectories = dict()
    trajectory = []
    states = []
    trajectories = []

    # model definition
    class Actor(torch.nn.Module):
        def __init__(self, state_dim, action_dim, action_range):
            super().__init__()

            self.fc1 = torch.nn.Linear(state_dim, 16)
            self.fc2 = torch.nn.Linear(16, 16)
            self.fc3 = torch.nn.Linear(16, action_dim)
            self.action_range = action_range

        def forward(self, state):
            a = torch.relu(self.fc1(state))
            a = torch.relu(self.fc2(a))
            a = torch.tanh(self.fc3(a)) * self.action_range
            return a

    class Critic(torch.nn.Module):
        def __init__(self, state_dim, action_dim):
            super().__init__()

            self.fc1 = torch.nn.Linear(state_dim + action_dim, 16)
            self.fc2 = torch.nn.Linear(16, 16)
            self.fc3 = torch.nn.Linear(16, 1)

        def forward(self, state, action):
            state_action = torch.cat([state, action], 1)
            q = torch.relu(self.fc1(state_action))
            q = torch.relu(self.fc2(q))
            q = self.fc3(q)
            return q

    actor = Actor(observe_dim, action_dim, action_range)
    actor_t = Actor(observe_dim, action_dim, action_range)
    critic = Critic(observe_dim, action_dim)
    critic_t = Critic(observe_dim, action_dim)

    ddpg = DDPG(
        actor, actor_t, critic, critic_t, torch.optim.Adam, torch.nn.MSELoss(reduction="sum")
    )

    episode, step, reward_fulfilled = 0, 0, 0
    smoothed_total_reward = 0
    list_ep_reward = []
    while True:
        episode += 1
        total_reward = 0
        terminal = False
        step = 0
        state = env.reset(); s=state * scale_factor
        if tuple(s) not in states: states.append(tuple(s))
        state = torch.tensor(state, dtype=torch.float32).view(1, observe_dim)
        episode_transitions = []

        while not terminal and step <= max_steps:
            step += 1
            with torch.no_grad():
                old_state = state
                # agent model inference
                action = ddpg.act_with_noise(
                    {"state": old_state}, noise_param=noise_param, mode=noise_mode
                )
                state, reward, terminal, _ = env.step(action.numpy()); s_ = state.flatten() * scale_factor
                if tuple(s_) not in states: states.append(tuple(s_))
                state = torch.tensor(state, dtype=torch.float32).view(1, observe_dim)
                total_reward += reward[0]

                #env.render()

                if action_state:
                    trajectory.append(tuple((s, action.numpy().flatten())))
                else:
                    trajectory.append(s)

                s = s_
                episode_transitions.append(
                    {
                        "state": {"state": old_state},
                        "action": {"action": action},
                        "next_state": {"state": state},
                        "reward": reward[0],
                        "terminal": terminal or step == max_steps,
                    }
                )
        ddpg.store_episode(episode_transitions)
        if action_state:
            trajectory.append(tuple((s_, action.numpy().flatten())))
            pass
        else:
            trajectory.append(s_)

        # update, update more if episode is longer, else less
        if episode > 100:
            for _ in range(step):
                ddpg.update()

        # show reward
        smoothed_total_reward = smoothed_total_reward * 0.9 + total_reward * 0.1
        logger.info(f"Episode {episode} total reward={smoothed_total_reward:.2f}")

        if episode > 100:
            trajectories.append(tuple(trajectory))
            list_ep_reward.append(total_reward)
            print("reward:", sum(list_ep_reward) / len(list_ep_reward))
            print("max_reward:", max(list_ep_reward))

        trajectory.clear()

        if len(trajectories) == n_of_trajectories: break

    return trajectories, states

def ddpg_mountain_car_collect_trajectories(env, n_of_trajectories=40, obs_noise=None, action_state = False, discrete = False, scale_factor=1.0):
    # configurations
    observe_dim = 3
    action_dim = 1
    action_range = 2
    max_steps = 200
    noise_param = (0, 0.2)
    noise_mode = "normal"

    dataset_trajectories = dict()
    trajectory = []
    states = []
    trajectories = []

    # model definition
    class Actor(torch.nn.Module):
        def __init__(self, state_dim, action_dim, action_range):
            super().__init__()

            self.fc1 = torch.nn.Linear(state_dim, 16)
            self.fc2 = torch.nn.Linear(16, 16)
            self.fc3 = torch.nn.Linear(16, action_dim)
            self.action_range = action_range

        def forward(self, state):
            a = torch.relu(self.fc1(state))
            a = torch.relu(self.fc2(a))
            a = torch.tanh(self.fc3(a)) * self.action_range
            return a

    class Critic(torch.nn.Module):
        def __init__(self, state_dim, action_dim):
            super().__init__()

            self.fc1 = torch.nn.Linear(state_dim + action_dim, 16)
            self.fc2 = torch.nn.Linear(16, 16)
            self.fc3 = torch.nn.Linear(16, 1)

        def forward(self, state, action):
            state_action = torch.cat([state, action], 1)
            q = torch.relu(self.fc1(state_action))
            q = torch.relu(self.fc2(q))
            q = self.fc3(q)
            return q

    actor = Actor(observe_dim, action_dim, action_range)
    actor_t = Actor(observe_dim, action_dim, action_range)
    critic = Critic(observe_dim, action_dim)
    critic_t = Critic(observe_dim, action_dim)

    ddpg = DDPG(
        actor, actor_t, critic, critic_t, torch.optim.Adam, torch.nn.MSELoss(reduction="sum")
    )

    episode, step, reward_fulfilled = 0, 0, 0
    smoothed_total_reward = 0

    while True:
        episode += 1
        total_reward = 0
        terminal = False
        step = 0
        state = env.reset(); s=state * scale_factor
        if tuple(s) not in states: states.append(tuple(s))
        state = torch.tensor(state, dtype=torch.float32).view(1, observe_dim)
        episode_transitions = []

        while not terminal and step <= max_steps:
            step += 1
            with torch.no_grad():
                old_state = state
                # agent model inference
                action = ddpg.act_with_noise(
                    {"state": old_state}, noise_param=noise_param, mode=noise_mode
                )
                state, reward, terminal, _ = env.step(action.numpy()); s_ = state.flatten() * scale_factor
                if tuple(s_) not in states: states.append(tuple(s_))
                state = torch.tensor(state, dtype=torch.float32).view(1, observe_dim)
                total_reward += reward[0]

                #env.render()

                if action_state:
                    trajectory.append(tuple((s, action.numpy().flatten())))
                else:
                    trajectory.append(s)

                s = s_
                episode_transitions.append(
                    {
                        "state": {"state": old_state},
                        "action": {"action": action},
                        "next_state": {"state": state},
                        "reward": reward[0],
                        "terminal": terminal or step == max_steps,
                    }
                )
        ddpg.store_episode(episode_transitions)
        if action_state:
            trajectory.append(tuple((s_, action.numpy().flatten())))
            pass
        else:
            trajectory.append(s_)

        # update, update more if episode is longer, else less
        if episode > 100:
            for _ in range(step):
                ddpg.update()

        # show reward
        smoothed_total_reward = smoothed_total_reward * 0.9 + total_reward * 0.1
        logger.info(f"Episode {episode} total reward={smoothed_total_reward:.2f}")

        if episode > 100:
            trajectories.append(tuple(trajectory))

        trajectory.clear()

        if len(trajectories) == n_of_trajectories: break

    return trajectories, states


def collect_FetchReach(env, n_of_trajectories, r_action_fn, max_steps):
    trajectories = []
    trajectory = []

    for _ in range(n_of_trajectories):
        state_info = env.reset()
        s = state_info["observation"]
        goal = state_info["desired_goal"]
        steps = 0
        done = False
        goal_reached_fn = lambda s, goal: np.linalg.norm(s - goal, 2) < 0.01
        while steps < max_steps or goal_reached_fn(s, goal):
            a = r_action_fn()[0]
            state_info, reward, done, info = env.step(a); steps += 1
            s_ = state_info["observation"]
            trajectory.append(tuple((s, copy.deepcopy(a))))
            s = s_
        trajectory.append(tuple((s, np.array([0, 0, 0]))))  # DO I NEED THIS OR NOT??? THIS WOULD BE A TERMINAL STATE!!!
        trajectories.append(tuple(trajectory))
        trajectory.clear()
    return trajectories


def collect_discrete_FetchReach(env, n_of_trajectories, max_steps):
    trajectories = []
    trajectory = []

    for _ in range(n_of_trajectories):
        action_space = list(np.array(np.meshgrid(np.arange(-1, 1.5, 0.5), np.arange(-1, 1.5, 0.5), np.arange(-1, 1.5, 0.5))).T.reshape(-1, 3))
        state_info = env.reset()
        s = state_info["observation"]
        goal = state_info["desired_goal"]
        steps = 0
        done = False
        goal_reached_fn = lambda s, goal: np.linalg.norm(s - goal, 2) < 0.01
        while steps < max_steps or goal_reached_fn(s, goal):
            a_i = random.choice(list(range(len(action_space))))
            state_info, reward, done, info = env.step(action_space[a_i]); steps += 1
            s_ = state_info["observation"]
            trajectory.append(tuple((s, copy.deepcopy(action_space[a_i]))))
            s = s_
        trajectory.append(tuple((s, np.array([0, 0, 0]))))  # DO I NEED THIS OR NOT??? THIS WOULD BE A TERMINAL STATE!!!
        trajectories.append(tuple(trajectory))
        trajectory.clear()
    return trajectories


def evaluate_point_mass(seed, max_n_steps_episode, agent):
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = PointmassEnv(max_n_steps_episode=max_n_steps_episode)
    env.seed(seed)

    for episode in range(10):
        s = env.reset()
        desired_goal = env.get_goal()
        done = False
        states = []
        actions = []
        while not done:
            states.append(s)
            a = np.random.choice(env.action_space.n)
            actions.append(a)
            s_, r, done, info = env.step(a)
            s = s_
        states.append(s)
        experience_replay.add_trajectory(states, actions, desired_goal)

def calculate_stats_dataset(trajectories):
    R = []
    for t in trajectories:
        R.append(-len(t))
    print("average ", sum(R)/len(R))
    print("max ", max(R))


def convert_traj_to_sas(trajectories: list) -> list:
    """
    A function to convert trajectories in s, a, s_ format
    Args:
        trajectories:

    Returns:
        trajectories_sas: trajectories in s, a, s_ format

    """
    trajectories_sas = []
    for traj in trajectories:
        traj_sas = []
        for i in range(len(traj) - 1):
            traj_sas.append({"s": traj[i][0], "a": traj[i][1], "s_": traj[i + 1][0]})

        trajectories_sas.append(traj_sas)

    return trajectories_sas

