import argparse
import gym
import importlib.util

parser = argparse.ArgumentParser()
parser.add_argument("--agentfile", type=str, help="file with Agent object", default="agents/qlearning_agent.py")
parser.add_argument("--env", type=str, help="Environment", default="FrozenLake-v1")
args = parser.parse_args()

spec = importlib.util.spec_from_file_location('Agent', args.agentfile)
agentfile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agentfile)
reward = []

try:
    env = gym.make(args.env)
    print("Loaded ", args.env)
except:
    print(args.env +':Env')
    gym.envs.register(
        id=args.env + "-v0",
        entry_point=args.env +':Env',
    )
    env = gym.make(args.env + "-v0")
    print("Loaded", args.env)

action_dim = env.action_space.n
state_dim = env.observation_space.n

agent = agentfile.Agent(state_dim, action_dim)

observation = env.reset() # representing the current state
rewards = []

max_reward, max_learning_rate, max_future_discount_factor = 0, 0, 0

lr = 0
for _ in range(10):
    if lr + 0.1 >= 1:
        lr = 0.1
    else:
        lr += 0.1
    agent.learning_rate = lr
    future_discount_factor = 0
    for _ in range(10):
        if future_discount_factor + 0.1 >= 1:
            future_discount_factor = 0.1
        else:
            future_discount_factor += 0.1
        agent.future_discount_factor = future_discount_factor
        for i in range (5):
            total_reward = 0
            for _ in range(125000): 
                # env.render()
                action = agent.act(observation) # your agent here (this takes random actions)
                observation, reward, done, info = env.step(action)
                agent.observe(observation, reward, done)
                total_reward += reward
                
                if done:
                    observation = env.reset() 
            rewards.append(total_reward)
        print(sum(rewards)/len(rewards), agent.learning_rate, agent.future_discount_factor)
        if sum(rewards)/len(rewards) > max_reward:
            max_reward = sum(rewards)/len(rewards)
            max_learning_rate = agent.learning_rate
            max_future_discount_factor = agent.future_discount_factor
            print("changed")
print(max_reward, max_learning_rate, max_future_discount_factor)
env.close()

