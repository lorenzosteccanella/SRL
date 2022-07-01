import random
import gym
from gym_minigrid.wrappers import NESWActionsImageDiagonal, NESWActionsImage
from matplotlib import pyplot as plt

env = gym.make("MiniGrid-Onlykey-10x10-v0")
env = NESWActionsImageDiagonal(env, max_num_actions=40)

s = env.reset()
done = False
while not done:
    a = random.choice(list(range(env.action_space.n)))
    s_, _, done, _ = env.step(4)
    plt.imshow(s_["image"])
    plt.show()