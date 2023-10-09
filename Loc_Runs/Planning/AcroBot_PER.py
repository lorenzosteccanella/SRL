import os
import sys
dirpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(dirpath))

from Planning.Planning_alg import *
from Problem_formulation.Models import LearnedModel
import gym

seed = 1

env = gym.make("Acrobot-v1")

env.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

action_space = list(range(env.action_space.n))
model = LearnedModel(6, 64, len(action_space))
model.load_state_dict(
    torch.load(dirname + "/Models/Acrobot_action_states_dictionary_2_"+str(seed)))
model.eval()
planner = Planner(model, list(range(env.action_space.n)), 20, 5)
desired_goal = [[-0.9661,  0.2581,  0.8875,  0.4607, -1.8354, -5.0000]]
desired_goal = planner.get_z_goal_states(desired_goal)


steps = 0

while steps < 100000:
    s = env.reset()
    tot_reward = 0.
    while True:
        p_a = planner.prob_multi_step_lookahead(s, desired_goal, 0.01, verbose=False)
        s_, r, done, info = env.step(p_a); steps += 1; tot_reward+=r
        s = s_
        #env.render()
        if done: print(steps, tot_reward, s); break