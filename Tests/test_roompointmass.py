from room_world.pointmass import PMEnv
import gym

config = dict(
    room_type="empty",
    potential_typ="none",
    shaped=True,
    max_path_len=200,
    use_state_images=False,
    use_goal_images=False
)

env = gym.make("Reacher-v2")

s = env.reset()
env.render()

for i in range(1000):
    env.step(0)
    env.render()