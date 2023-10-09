import os
import sys
dirpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(dirpath))

from Planning.Planning_alg import *
from Problem_formulation.Models import LearnedModel
import gym
from multiworld import *

register_all_envs()

seed = 1

env = gym.make("SawyerReachXYZEnv-v1")

env.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

action_space = list(np.array(
  np.meshgrid(np.arange(-1, 1.5, 0.5), np.arange(-1, 1.5, 0.5), np.arange(-1, 1.5, 0.5))).T.reshape(-1, 3))
model = LearnedModel(3, 64, 3)
model.load_state_dict(
    torch.load(dirname + "/Models/FetchReach_action_states_dictionary_disc_2_"+str(seed)))
model.eval()
planner = Planner(model, action_space, 10, 1)


steps = 0

max_steps = 200
tot_steps = 0
while tot_steps < 100000:
    state_info = env.reset()
    s = state_info["observation"]
    desired_goal = state_info["desired_goal"]
    z_goals = planner.get_z_goal_states([desired_goal])
    steps = 0
    goal_reached_fn = lambda s, goal: not np.linalg.norm(s-goal, 2) < 0.01
    while steps < max_steps and goal_reached_fn(s, desired_goal):
        p_a = planner.cont_prob_multi_step_lookahead(s, z_goals, 0.01, verbose=False)
        state_info, reward, done, info = env.step(p_a); steps += 1; tot_steps+=1
        s_ = state_info["observation"]
        s = s_
    print(s, desired_goal, np.linalg.norm(s - desired_goal, 2), goal_reached_fn(s, desired_goal))
