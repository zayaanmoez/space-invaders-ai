import gym
# from ale_py import ALEInterface
# import ale_py.roms as roms
# print(roms.__all__)
#
# ale = ALEInterface()
# ale.loadROM(space_invaders)

env = gym.make('ALE/SpaceInvaders-v5', render_mode='human')
env.reset()

for _ in range(1000):
    env.step(env.action_space.sample())