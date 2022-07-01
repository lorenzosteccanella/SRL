import itertools
import random
import numpy as np
import torch
from Problem_formulation.minibatch_deep_d_PER import LearnedModel
import gym
from icecream import ic


def density_space(xs, ps, n, endpoint=False, order=1, random=False):
    """Draw samples with spacing specified by a density function.

    Copyright Han-Kwang Nienhuys (2020).
    License: any of: CC-BY, CC-BY-SA, BSD, LGPL, GPL.
    Reference: https://stackoverflow.com/a/62740029/6228891

    Parameters:

    - xs: array, ordered by increasing values.
    - ps: array, corresponding densities (not normalized).
    - n: number of output values.
    - endpoint: whether to include x[-1] in the output.
    - order: interpolation order (1 or 2). Order 2 will
      require dense sampling and a smoothly varying density
      to work correctly.
    - random: whether to return random samples, ignoring endpoint).
      in this case, n can be a shape tuple.

    Return:

    - array, shape (n,), with values from xs[0] to xs[-1]
    """
    from scipy.interpolate import interp1d
    from scipy.integrate import cumtrapz

    cps = cumtrapz(ps, xs, initial=0)
    cps *= (1 / cps[-1])
    intfunc = interp1d(cps, xs, kind=order)
    if random:
        return intfunc(np.random.uniform(size=n))
    else:
        return intfunc(np.linspace(0, 1, n, endpoint=endpoint))

env = gym.make("Pendulum-v0")
action_space = density_space(
    [-2, 0, 2],
    [0.1, 0.8, 0.1],
    n=10, endpoint=True)
encoder = LearnedModel(3, 32, 1)
encoder.load_state_dict(
    torch.load("/home/lorenzo/Documenti/UPF/MDP_temporal_representation/Models/Pendulum_action_states_dictionary"))
encoder.eval()
z_goal_states = []
z_goal_states.append(encoder.encode_state(torch.FloatTensor([[1, 0, 0.]])))

# set seed
env.seed(1)
torch.manual_seed(1)
np.random.seed(1)
random.seed(1)

steps = 0

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

def action_step_lookahead(model, z, action_traj, actions, horizon, dist):
    for a, step in zip(action_traj, range(horizon)):
        z_a = torch.FloatTensor([[a]])
        z_ = model.encode_action(z, z_a)
        #if step == horizon-1:
        dist += avg_dist_to_goal_states(model, z_, z_goal_states)
        z = z_
    return dist

def multi_step_lookahead(model, s, actions, max_n_rand_traj=10):
    horizon = 5
    action_traj_comb = list(itertools.combinations_with_replacement(actions, horizon))
    action_traj_comb = random.sample(action_traj_comb, min(len(action_traj_comb), max_n_rand_traj))
    actions_values = np.ones(len(actions)) * float("inf")
    s = torch.FloatTensor(np.array([s]))
    for index_a, a in enumerate(actions):
        z_a = torch.FloatTensor([[a]])
        z_ = model.encode_next_state(s, z_a)
        dist = avg_dist_to_goal_states(model, z_, z_goal_states)
        for action_traj in action_traj_comb:
            assert len(action_traj) > 0
            dist = action_step_lookahead(model, z_, action_traj, actions, horizon, dist)
            actions_values[index_a] = min(dist, actions_values[index_a])#(dist/(len(action_traj_comb)+1))
    return actions[np.argmin(actions_values)]

while steps < 50000:
    s = np.array(env.reset())
    tot_reward = 0.
    tot_pseudo_reward = 0.
    while True:
        shortest_path_action = multi_step_lookahead(encoder, s, action_space)
        s_, r, done, info = env.step(np.array([shortest_path_action])); steps += 1; tot_reward+=r
        s = s_
        env.render()
        if done: print(steps, tot_reward, s); break