import os
import sys
dirpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(dirpath))

from Baselines.GCSL import GCSL, GCSL_experience_replay, GCSL_NN
from Utils.Utils import *
from multiworld import *
import gym

register_all_envs()

seed = 1

env = gym.make("SawyerReachXYZEnv-v1")
max_n_steps_episode = 200
action_space = list(np.array(
  np.meshgrid(np.arange(-1, 1.5, 0.5), np.arange(-1, 1.5, 0.5), np.arange(-1, 1.5, 0.5))).T.reshape(-1, 3))

env.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

model = GCSL_NN(max_n_steps_episode=max_n_steps_episode, input_size= 6, actions=action_space)
experience_replay = GCSL_experience_replay()
optimizer = torch.optim.Adam
learning_rate = 0.0005
agent = GCSL(model, optimizer, learning_rate)

trajectories = load_action_trajectories(dirname + "/Data/trajectory_FetchReach_disc_2.npy")
experience_replay.load_trajectories(trajectories, None)

# train the agent
n_steps = 100000
for i in range(n_steps):
    s_batch, g_batch, h_batch, a_batch = experience_replay.sample_batch(batch_size=256)
    agent.train(s_batch, g_batch, None, a_batch)
    if i%100==0: print(i)

wandb_group = "FetchReach_GCSL" + "_wo_h"
wandb_name = "FetchReach_GCSL_seed_" + str(seed) + "_wo_h"

run = wandb.init(project='MDP_DIST', entity='lsteccanella',
                 group=wandb_group, settings=wandb.Settings(start_method="fork"))
wandb.run.name = wandb_name

steps = 0
while steps < 100000:
    state_info = env.reset()
    s = state_info["observation"]
    desired_goal = state_info["desired_goal"]
    tot_reward = 0.
    i = 0
    ep_steps=0
    goal_reached_fn = lambda s, goal: not np.linalg.norm(s-goal, 2) < 0.01
    while ep_steps < max_n_steps_episode and goal_reached_fn(s, desired_goal):
        a = agent.act(state=s, goal=desired_goal, horizon=None)
        state_info, r, done, info = env.step(a); ep_steps += 1; steps+=1
        s_ = state_info["observation"]
        s = s_
        #env.render()

    print({"dist_goal": np.linalg.norm(s - desired_goal, 2),
               "n_steps": ep_steps})

