import numpy as np

class Agent(object): #Keep the class name!
    """The world's simplest agent!"""
    def __init__(self, state_space, action_space):
        self.action_space = action_space
        self.state_space = state_space
        #Added code
        self.learning_rate = 0.1
        self.future_discount_factor = 0.9
        self.epsilon = 0.05
        self.Q = 0.5*np.ones((state_space, action_space))
        
    def observe(self, observation, reward, done):
        #Add your code here
        self.s_prime = observation #not really necessary
        if not done :
            #print("Not done yet")
            self.Q[self.s,self.a] += self.alpha*(reward+self.gamma*np.argmax(self.Q[self.s_prime, :])-self.Q[self.s,self.a])
            self.s = self.s_prime
            #self.a = self.a_prime  #maybe for SARSA
        #else:
            #print("end of episode")
                
        pass
    
    def act(self, observation):
        #Add your code here
        self.s = observation  #not really necessary
        if np.random.rand() > self.epsilon: 
            action = np.argmax(self.Q[self.s,:]) 
        else: 
            action = np.random.randint(self.action_space) 
        #print("action selected", action)
        self.a = action
        return action

        #Commented out by us
        #return np.random.randint(self.action_space)