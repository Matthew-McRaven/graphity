import torch

import graphity.environment.reward, graphity.environment.sim
import graphity.agent.markov

size = 10
agent = graphity.agent.markov.MDPAgent()
env = graphity.environment.sim.Simulator()

state=torch.ones((size, size), dtype=torch.int64)

# Blank out the diagonal. Don't know how self-loops work.
for i in range(size):
    state[i,i] = 0

env.reset(state)
count, reward = 0, float('inf')

while reward > 0:
    count+=1
    action = agent(state)
    state, reward = env.step(action)

print(count, reward.item())
print(state)