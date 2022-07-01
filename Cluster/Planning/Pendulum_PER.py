import os
import sys
dirpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dirname = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(dirpath))

from Planning.Planning_alg import *
from Problem_formulation.Models import LearnedModel
import gym

seed = int(sys.argv[1])

env = gym.make("Pendulum-v0")

env.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

action_space = list(np.array(np.arange(-2, 2.4, 0.4)).reshape(-1, 1))
print(action_space)
model = LearnedModel(3, 64, 1)
model.load_state_dict(
    torch.load(dirname + "/Models/Pendulum_action_states_dictionary_3_"+str(seed)))
model.eval()
planner = Planner(model, action_space, 10, 5)
desired_goal = [[1, 0, 0]]
desired_goal = planner.get_z_goal_states(desired_goal)


steps = 0

wandb_group = "Pendulum_Planning"
wandb_name = "Pendulum_Planning_seed_" + str(seed)

run = wandb.init(project='MDP_DIST', entity='lsteccanella',
                 group=wandb_group, settings=wandb.Settings(start_method="fork"))
wandb.run.name = wandb_name

while steps < 100000:
    s = env.reset().flatten()
    tot_reward = 0.
    while True:
        p_a = planner.cont_prob_multi_step_lookahead(s, desired_goal, 0.01, verbose=False)
        s_, r, done, info = env.step(np.array([p_a])); steps += 1; tot_reward+=r
        s = s_.flatten()
        #env.render()
        if done: print(steps, tot_reward, s); break

    wandb.log({"reward": tot_reward},
              step=steps)
