import os
import sys
dirpath = os.path.dirname(os.path.abspath(__file__))
dirname = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(dirpath))

from Envs.PointmassEnv import PointmassEnv
import numpy as np
import torch
from Baselines.GCSL import GCSL, GCSL_experience_replay, GCSL_NN

seed = 1
np.random.seed(seed)
torch.manual_seed(seed)
env = PointmassEnv(max_n_steps_episode=50)
env.seed(seed)

model = GCSL_NN(max_n_steps_episode=env.max_n_steps_episode, actions=list(range(env.action_space.n)))
experience_replay = GCSL_experience_replay()
optimizer = torch.optim.Adam
learning_rate = 0.0005
agent = GCSL(model, optimizer, learning_rate)

for episode in range(1000):
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

# train the agent
n_steps = 100000
for i in range(n_steps):
    s_batch, g_batch, h_batch, a_batch = experience_replay.sample_batch(batch_size=256)
    agent.train(s_batch, g_batch, h_batch, a_batch)
    if i%100==0: print(i)


# evaluate the agent
seed = 1
np.random.seed(seed)
torch.manual_seed(seed)
env = PointmassEnv(max_n_steps_episode=50)
env.seed(seed)

for episode in range(100):
    s = env.reset()
    desired_goal = env.get_goal()
    done = False
    states = []
    actions = []
    for i in range(env.max_n_steps_episode):
        states.append(s)
        a = agent.act(state=s, goal=desired_goal, horizon=np.array(env.max_n_steps_episode-i, dtype=float))
        actions.append(a)
        s_, r, done, info = env.step(a)
        s = s_
        if done: break
    print(desired_goal, s)




