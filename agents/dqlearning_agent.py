import numpy as np
import random

class Agent(object): #Keep the class name!
    """The world's simplest agent!"""
    def __init__(self, state_space, action_space):
        self.action_space = action_space
        self.state_space = state_space
        
        self.learning_rate = 0.1
        self.future_discount_factor = 1
        self.qa_table = np.ones([self.state_space, self.action_space])
        self.qb_table = np.ones([self.state_space, self.action_space])
        
        self.epsilon = 0.05
        self.latest_state = None


    def observe(self, observation, reward, done):
        if random.uniform(0,1) > 0.5:
            a_star = np.argmax(self.qa_table[observation]) #hmmm
            future = (self.future_discount_factor * self.qb_table[observation, a_star] - self.qa_table[self.latest_state,self.action] )if not done else 0
            self.qa_table[self.latest_state, self.action] = self.qa_table[self.latest_state,self.action] + self.learning_rate * \
                (reward + future)
        else:
            a_star = np.argmax(self.qb_table[observation]) #hmmm
            future = (self.future_discount_factor * self.qa_table[observation, a_star] - self.qb_table[self.latest_state,self.action] )if not done else 0
            self.qb_table[self.latest_state, self.action] = self.qb_table[self.latest_state,self.action] + self.learning_rate * \
                (reward + future )    

 
    def act(self, observation):
        self.latest_state = observation
        #print(self.q_table)
        if random.uniform(0, 1) <= self.epsilon:
            self.action = np.random.randint(self.action_space) # Explore
        else:
            self.action = np.argmax(self.qa_table[observation] + self.qa_table[observation]) # Exploit

        return self.action
