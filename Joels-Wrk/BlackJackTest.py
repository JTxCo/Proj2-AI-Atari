

from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from tqdm import tqdm

import gymnasium as gym


env = gym.make('Blackjack-v1', natural=False, sab=False)
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


# class BlackjackAgent:
#     def __init__(
#         self,
#         learning_rate: float,
#         initial_epsilon: float,
#         epsilon_decay: float,
#         final_epsilon: float,
#         discount_factor: float = 0.95,
#     ):
#         self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
#         self.lr = learning_rate
#         self.discount_factor = discount_factor
#         self.epsilon = epsilon_decay
#         self.epsilon_decay = epsilon_decay
#         self.final_epsilon = final_epsilon
        
#         self.training_error = []
        
    
#     def get_action(self, obs: tuple[int, int, bool]) -> int:
#         if np.random.random() < self.epsilon:
#             return env.action_space.sample()
#         else:
#             return np.argmax(self.q_values[obs])
        
#     def update(
#         self,
#         obs: tuple[int, int, bool],
#         action: int,
#         reward: float,
#         terminated: bool,
#         next_ob: tuple[int, int, bool],
        
#     ):
#         future_q_value = (not terminated) * np.max(self.q_values[next_ob])
#         temporal_difference = reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        
#         self.q_values[obs][action] += self.lr * temporal_difference
#         self.training_error.append(temporal_difference)
        
#     def decay_epsilon(self):
#         self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)
        
        
# learning_rate = 0.1
# n_episodes = 100_000
# start_epsilon = 1.0
# epsilon_decay = start_epsilon/ (n_episodes /2)
# final_epsilon = 0.1

# agent = BlackjackAgent(
#     learning_rate=learning_rate,
#     initial_epsilon=start_epsilon,
#     epsilon_decay=epsilon_decay,
#     final_epsilon=final_epsilon,
# )

class BlackjackAgent:

    def __init__(self, env: gym.BlackjackEnv, alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: np.zeros(env.action_space.n))

    def policy(self, state: tuple, epsilon: float = None) -> int:
        if epsilon is None:
            epsilon = self.epsilon
        if np.random.random() < epsilon:
            return np.random.choice(self.env.action_space.n)
        else:
            return np.argmax(self.Q[state])

    def train(self, num_episodes: int = 100000, verbose: bool = False) -> list:
        rewards = []
        for i in tqdm(range(num_episodes)):
            done = False
            observation, info = self.env.reset()
            while not done:
                action = self.policy(observation)
                observation_, reward, done, _, _ = self.env.step(action)
                self.Q[observation][action] += self.alpha * (reward + self.gamma * np.max(self.Q[observation_]) - self.Q[observation][action])
                observation = observation_
            rewards.append(reward)
            if verbose and i % 10000 == 0:
                print(f"Episode {i} finished with reward {reward}")
        return rewards

    def play(self, num_episodes: int = 100) -> list:
        rewards = []
        for i in range(num_episodes):
            done = False
            observation, info = self.env.reset()
            while not done:
                action = self.policy(observation, epsilon=0)
                observation_, reward, done, _, _ = self.env.step(action)
                observation = observation_
            rewards.append(reward)
        return rewards

