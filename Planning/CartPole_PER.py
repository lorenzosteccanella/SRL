import itertools
import random
import time
import numpy as np
import torch
import wandb

from Problem_formulation.Models import LearnedModel
import gym

env = gym.make("CartPole-v0")
action_space = list(range(env.action_space.n))
model = LearnedModel(4, 64, len(action_space))
model.load_state_dict(
    torch.load("/home/lorenzo/Documenti/UPF/MDP_temporal_representation/Models/CartPole_action_states_dictionary"))
model.eval()
z_goal_states = []
# for pos in range(-10, +10+1, 1):
#     # for cart_velocity in range(-2, 2+1, 1):
#     #     for pole_angle in range(-1, 1+1, 1):
#     #         for pole_velocity in range(-2, 2+1, 1):
#     #             z_goal_states.append(model.encode_state(torch.FloatTensor([[0.1 * pos, cart_velocity,  0.1*pole_angle, pole_velocity]])))
#     z_goal_states.append(model.encode_state(torch.FloatTensor([[0.01 * pos, 0,  0, 0]])))
z_goal_states.append(model.encode_state(torch.FloatTensor([[0, 0, 0, 0]])))

# set seed
seed = 1
env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

steps = 0

# wandb_group = "CartPole_Planning"
# wandb_name = "CartPole_Planning_seed_" + str(seed)
#
# run = wandb.init(project='MDP_DIST', entity='lsteccanella',
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

def one_hot_action(actions, a):
    one_hot_a = np.zeros(len(actions))
    one_hot_a[a] = 1
    one_hot_a = np.expand_dims(one_hot_a, axis=0)
    one_hot_a = torch.FloatTensor(one_hot_a)
    return one_hot_a

def action_step_lookahead(model, z, action_traj, actions, horizon, dist):
    for a, step in zip(action_traj, range(horizon)):
        one_hot_a = one_hot_action(actions, a)
        z_ = model.encode_action(z, one_hot_a)
        # if step == horizon-1:
        #     dist = min_dist_to_goal_states(model, z_, z_goal_states)
        dist += min_dist_to_goal_states(model, z_, z_goal_states)
        z = z_
    return dist

def multi_step_lookahead(model, s, actions, max_n_rand_traj=10):
    horizon = 5
    action_traj_comb = list(itertools.combinations_with_replacement(actions, horizon))
    action_traj_comb = random.sample(action_traj_comb, min(len(action_traj_comb), max_n_rand_traj))
    actions_values = np.ones(len(actions)) * float("inf")
    s = torch.FloatTensor(np.array([s]))
    for a in actions:
        one_hot_a = one_hot_action(actions, a)
        z_ = model.encode_next_state(s, one_hot_a)
        dist = min_dist_to_goal_states(model, z_, z_goal_states)
        for action_traj in action_traj_comb:
            assert len(action_traj) > 0
            dist = action_step_lookahead(model, z_, action_traj, actions, horizon, dist)
            actions_values[a] = min(dist, actions_values[a])#(dist/(len(action_traj_comb)+1))
    return np.argmin(actions_values)


while steps < 20000:
    s = env.reset()
    tot_reward = 0.
    tot_pseudo_reward = 0.
    while True:
        shortest_path_action = multi_step_lookahead(model, s, action_space)
        s_, r, done, info = env.step(shortest_path_action); steps += 1; tot_reward+=r
        s = s_
        env.render()
        if done: print(steps, tot_reward, s); break

    # wandb.log({"reward": tot_reward},
    #           step=steps)