# To create a Q-learning model for the BattleZone game, we will need to define the following:
# 1.	Objective of the game: The objective of the game is to control a tank and destroy enemy vehicles while avoiding being hit by enemy fire.
# 2.	Actions that the agent can take: The agent can take 18 different actions, such as moving up, down, left, or right, firing, or combinations of these actions.
# 3.	States that the agent can be in: The agent's state is defined by the observations it receives from the game, which consist of a 210x160x3 RGB image of the game screen.
# 4.	Reward structure of the game: The agent receives points for destroying enemy vehicles.
# 
#       With this information, we can create a Q-learning agent
#       that learns to play the BattleZone game. We can start by defining a Q-learning class that initializes the Q-values for each state-action pair, and implements the Q-learning algorithm to 
#       update the Q-values based on the rewards received and the next state.Here's an example code snippet to get you started:
    
import gym
import numpy as np

# Define the Q-learning agent class
class QLearningAgent:
    def __init__(self, state_size, action_size, alpha, gamma, epsilon):
        self.q_table = np.zeros((state_size, action_size))
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def act(self, state):
        if np.random.uniform() < self.epsilon:
            return np.random.choice(range(self.q_table.shape[1]))
        else:
            return np.argmax(self.q_table[state, :])

    def learn(self, state, action, reward, next_state, done):
        q_next = np.max(self.q_table[next_state, :])
        q_target = reward + self.gamma * q_next * (1 - done)
        q_current = self.q_table[state, action]
        self.q_table[state, action] += self.alpha * (q_target - q_current)

# Create the environment
env = gym.make('ALE/BattleZone-v5', render_mode = "human")
env.metadata['render_fps'] = 60  # set the render fps to 60
# Create the agent
agent = QLearningAgent(state_size=env.observation_space.shape[0],
                       action_size=env.action_space.n,
                       alpha=0.1,
                       gamma=0.99,
                       epsilon=1.0)

# Train the agent
for episode in range(10000):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        env.render()  # display the current state of the game
        # print("state:", state)
        action = agent.act(state)
        # print("action:", action)
        actions = env.step(action)
        # print(f"\n actions: {actions}\n")
        if isinstance(actions, tuple) and len(actions) == 5:
            next_state, reward, done, info1, info2 = actions
            my_tuple = actions
            my_list  = [str(next_state), str(reward), str(done), str(info1), str(info2)]
            # for i in range(len(my_tuple)):
            #     print(f"{my_list [i]}:   {my_tuple[i]}\n")
            
        else:
            raise ValueError("Invalid tuple returned by env.step()")
        agent.learn(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
    print(f"Episode: {episode}, Total Reward: {total_reward}")
    
    
    
    # # Train the agent
    # for episode in range(10000):
    #     state = env.reset()
    #     done = False
    #     total_reward = 0
    #     while not done:
    #         env.render()  # display the current state of the game
    #         action = agent.act(state)
    #         actions = env.step(action)
    #         if isinstance(actions, tuple) and len(actions) == 4:
    #             next_state, reward, done, info = actions
    #         elif isinstance(actions, tuple) and len(actions) == 3:
    #             next_state, reward, done = actions
    #             info = None
    #         else:
    #             raise ValueError("Invalid tuple returned by env.step()")
    #         agent.learn(state, action, reward, next_state, done)
    #         state = next_state
    #         total_reward += reward
    #     print(f"Episode: {episode}, Total Reward: {total_reward}")