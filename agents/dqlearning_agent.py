import numpy as np
import random

class Agent(object): #Keep the class name!
    """The world's simplest agent!"""
    def __init__(self, state_space, action_space):
        self.action_space = action_space
        self.state_space = state_space
        
        self.learning_rate = 0.4
        self.future_discount_factor = 0.9
        self.qa_table = 0.5*np.ones([self.state_space, self.action_space])
        self.qb_table = 0.5*np.ones([self.state_space, self.action_space])
        
        self.epsilon = 0.05
        self.latest_state = None
        self.latest_action = None


    def observe(self, observation, reward, done):
        if random.uniform(0,1) > 0.5: # randomly choose either table a or b
            a_star = np.argmax(self.qa_table[observation])
            if done:
                potential_future_value = 0
            else: 
                potential_future_value = self.future_discount_factor * self.qb_table[observation, a_star] \
                    - self.qa_table[self.latest_state,self.latest_action]
            self.qa_table[self.latest_state, self.latest_action] = self.qa_table[self.latest_state,self.latest_action] \
                + self.learning_rate * (reward + potential_future_value)
        else:
            a_star = np.argmax(self.qb_table[observation])
            potential_future_value = self.future_discount_factor * self.qa_table[observation, a_star] \
                - self.qb_table[self.latest_state,self.latest_action] 
            self.qb_table[self.latest_state, self.latest_action] = self.qb_table[self.latest_state,self.latest_action] \
                + self.learning_rate * (reward + potential_future_value)    

 
    def act(self, observation):
        self.latest_state = observation
        if random.uniform(0, 1) <= self.epsilon:
            self.latest_action = np.random.randint(self.action_space) # Explore
        else:
            self.latest_action = np.argmax(self.qa_table[observation] + self.qa_table[observation]) # Exploit

        return self.latest_action
