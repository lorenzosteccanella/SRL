import copy
import random
import numpy as np
import wandb
from matplotlib import pyplot as plt
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

