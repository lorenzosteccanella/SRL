import os
import sys
dirpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(dirpath))

import numpy as np
import torch
import random
from Baselines.GCSL import GCSL, GCSL_experience_replay, GCSL_NN
import gym
from Utils.Utils import *

seed = int(sys.argv[1])

env = gym.make("Pendulum-v0")
actions = list(np.array(np.arange(-2, 2.1, 0.1)).reshape(-1, 1))
max_n_steps_episode = 200

env.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

model = GCSL_NN(max_n_steps_episode=max_n_steps_episode, input_size=7, actions=actions)
experience_replay = GCSL_experience_replay()
optimizer = torch.optim.Adam
learning_rate = 0.0005
agent = GCSL(model, optimizer, learning_rate)

trajectories = load_action_trajectories(dirname + "/Data/trajectory_Pendulum.npy")
desired_goal = np.array([1, 0, 0])
experience_replay.load_trajectories(trajectories, desired_goal, discretize_action=True, actions_list=actions)

# train the agent
n_steps = 100000
for i in range(n_steps):
    s_batch, g_batch, h_batch, a_batch = experience_replay.sample_batch(batch_size=256)
    agent.train(s_batch, g_batch, h_batch, a_batch)
    if i%100==0: print(i)

wandb_group = "Pendulum_GCSL" + "_0.1d"
wandb_name = "Pendulum_GCSL_seed_" + str(seed) + "_0.1d"

run = wandb.init(project='MDP_DIST', entity='lsteccanella',
                 group=wandb_group, settings=wandb.Settings(start_method="fork"))
wandb.run.name = wandb_name

steps = 0
while steps < 100000:
    s = env.reset()
    tot_reward = 0.
    i = 0
    while True:
        a = agent.act(state=s, goal=desired_goal, horizon=np.array(max_n_steps_episode-i, dtype=float))
        s_, r, done, info = env.step(a); steps += 1; tot_reward+=r; i+=1
        s = s_
        #env.render()
        if done: print(steps, tot_reward, s); break

    wandb.log({"reward": tot_reward},
              step=steps)

