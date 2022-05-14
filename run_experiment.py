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

aggregated_rewards = {}
max_reward, max_id = 0, 0
lr = 0
for _ in range(10): # for 10 different values of the learning rate (0.1 to 1 in increments of 0.1)
    lr = 0.1 if lr + 0.1 >= 1 else lr + 0.1
    agent.learning_rate = lr
    fdf = 0
    for _ in range(10): # for 10 different values of the future discount factor (0.1 to 1 in increments of 0.1)
        rewards = []
        fdf = 0.1 if fdf + 0.1 >= 1 else fdf + 0.1
        agent.future_discount_factor = fdf
        for _ in range(5): # runs
            total_reward_gained = 0
            for _ in range(125000): # steps (if this goes to infinity we get the optimal q-table)
                # env.render()
                action = agent.act(observation) # your agent here (this takes random actions)
                observation, reward, done, info = env.step(action)
                agent.observe(observation, reward, done)

                total_reward_gained += reward
                
                if done:
                    observation = env.reset() 
            rewards.append(total_reward_gained)
        i = '(' + str(round(lr, 2)) + ', ' + str(round(fdf, 2)) + ')'
        aggregated_rewards[i] = rewards
        average_reward = sum(aggregated_rewards[i])/len(aggregated_rewards[i])
        print(i, average_reward)

        if average_reward > max_reward:
            max_reward = average_reward
            max_id = i
            print("Current best changed to: " + i)
            
print(max_reward, max_id)
env.close()

