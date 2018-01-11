# Q-learning可以用tables（查找表）或者neural network（状态的规模无限大） 来实现。

"""
Q-table:
          action 
state a1 a2 a3 ...
   s1
   s2
   s3
   ...

jList = [] # steps per episode 每一轮有多少步
rList = [] # total rewards 奖励总和

for i in range(num_episodes): # 对于每一轮episode
    # 先reset环境
    while j < 99: # 当步数达到100， 或者达到目标后， 一轮结束
        # 选择动作
        # 得到环境反馈
        # 更新Q_table

    jList.append(j)
    rList.append(rAll)

# 可以求每一轮reward的平均值，做一个记录

1. reset environment and get first new observation
   s = env.reset()
   一开始，先重置环境，获取第一个新的观测

2. choose an action by greedily (with noise) picking from Q table
   根据观测，在Q table中选择一个动作
   a = np.argmax(Q[s, :] + np.random.randn(1, env.action_space.n) * (1./(i+1)))
   一般情况下，选择Q值最大的那个动作，但是这里会增加一些噪声，并且噪声会随着轮数的增多而减少，
   
3. get new state and reward from environment
   动作作用于环境，得到反馈信息，下一个状态，奖励，是否结束等等
   s1, r, d, _ = env.step(a)

4. update Q-table with new knowledge
   利用反馈信息更新Q-table，也就是更新Q值
   Q[s, a] = Q[s, a] + lr * (r + y * np.max(Q[s1, :]) - Q[s, a])
"""

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
