import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from ale_py import ALEInterface
ale = ALEInterface()

from ale_py.roms import SpaceInvaders
ale.loadROM(SpaceInvaders)

env = gym.make("SpaceInvaders-v4")
n_outputs = env.action_space.n
print(n_outputs)
print(env.env.get_action_meanings())
observation = env.reset()

for i in range(25):
    if i > 20:
        plt.imshow(observation)
        plt.show()
    
    observation, _, _, _, _ = env.step(1)