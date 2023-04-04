

from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from tqdm import tqdm

import gymnasium as gym


gym.make('Blackjack-v1', natural=False, sab=False)
done = False
observation, info = env.reset()

# observation = (16, 9, False) mesan (sum, dealer value, usable ace)


# sample a random action from all valid actions

action = env.action_space.sample()
# action = 1

# execute the action in our environment and receive infos from the environment
# sample a random action from all valid actions
action = env.action_space.sample()
# action=1

# execute the action in our environment and receive infos from the environment
observation, reward, terminated, truncated, info = env.step(action)

# observation=(24, 10, False)
# reward=-1.0
# terminated=True
# truncated=False
# info={} (empty dict) that holds info for lives, not this one

# building Q-learning agent for blackjack v1


class BlackjackAgent:
    def __init__(
        self,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
