import numpy as np
import random

class Agent(object): #Keep the class name!
    """The world's simplest agent!"""
    def __init__(self, state_space, action_space):
        self.action_space = action_space
        self.state_space = state_space

        self.learning_rate = 0.8
        self.future_discount_factor = 0.95
        self.qtable = 0.5*np.ones([self.state_space, self.action_space])

        self.epsilon = 0.05

        self.latest_state = None
        self.latest_action = None
                
        
    # observation is the current state
    def observe(self, observation, reward, done):
        if done:
            potential_future_reward = 0
        else:
            potential_future_reward = np.max(self.qtable[observation])
        old_Qvalue = self.qtable[self.latest_state, self.latest_action]
        self.qtable[self.latest_state, self.latest_action] = (1 - self.learning_rate) * old_Qvalue + self.learning_rate * (reward + self.future_discount_factor * potential_future_reward)

        
    def act(self, observation):
        self.latest_state = observation
        if random.uniform(0,1) <= self.epsilon:
            self.latest_action = np.random.randint(self.action_space)
        else:
            self.latest_action = np.argmax(self.qtable[observation])

        return self.latest_action