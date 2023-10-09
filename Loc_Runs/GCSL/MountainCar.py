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

seed = 1

env = gym.make("MountainCar-v0")
max_n_steps_episode = 200

env.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

model = GCSL_NN(max_n_steps_episode=max_n_steps_episode, input_size= 5, actions=list(range(env.action_space.n)))
experience_replay = GCSL_experience_replay()
optimizer = torch.optim.Adam
learning_rate = 0.0005
agent = GCSL(model, optimizer, learning_rate)

trajectories = load_action_trajectories(dirname + "/Data/trajectory_MountainCar.npy")
desired_goal = np.array([0.50427865, 0.02712902])
experience_replay.load_trajectories(trajectories, desired_goal)

# train the agent
n_steps = 100000
for i in range(n_steps):
    s_batch, g_batch, h_batch, a_batch = experience_replay.sample_batch(batch_size=256)
    agent.train(s_batch, g_batch, h_batch, a_batch)
    if i%100==0: print(i)

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

