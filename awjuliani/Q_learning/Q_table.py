import gym
import numpy as np 
import random
import matplotlib.pyploy as plt 

# %matplotlib inline

# # load the environment
env = gym.make('FronzenLake-v0')

# # implement Q-Table learning algorithm
# initialize table with all zeros
Q = np.zeros([env.observation_space.n, env.action_space.n])
# set learning parameters
lr = .8
y = .95
num_episodes = 2000
# create lists to contain total rewards and steps per episode
# jList = []
rList = []
for i in range(num_episodes):
	# reset environment and get first new observation
	s = env.reset()
	rAll = 0
	d = False
	j = 0
	# the Q-table learning algorithm
	while j < 99:
		j += 1
		# choose an action by greedily (with noise) picking from Q table
		a = np.argmax(Q[s, :] + np.random.randn(1, env.action_space.n) * (1./(i+1)))
		# get new state and reward from environment
		s1, r, d, _ = env.step(a)
		# update Q-table with new knowledge
		Q[s, a] = Q[s, a] + lr * (r + y * np.max(Q[s1, :]) - Q[s, a])
		rAll += r
		s = s1
		if d == True:
			break
	# jList.append(j)
	rList.append(rAll)

print("Score over time:" + str(sum(rList) / num_episodes))
print("Final Q-Table Values")	
print(Q)
