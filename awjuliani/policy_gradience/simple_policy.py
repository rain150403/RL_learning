# a simple example of how to build a policy-gradient based agent that can solve the multi-armed bandit problem.
# 基于策略梯度解决多臂赌博机问题

import tensorflow as tf 
import tensorflow.contrib.slim as slim
import numpy as np 

# the bandit

"""
here we define our bandit. for this example we are using a four-armed bandit. the pullBandit function generates a random number
from a normal distribution with a mean of 0. the lower the bandit number, the more likely a positive reward will be returned.
we want our agent to learn to always choose the arm that will give that positive reward.

在这里我们定义赌博机。对于这个例子我们用四臂赌博机。函数pullBandit（）从一个零均值的正态分布产生一个随机数字。赌博机的数字越小，就越可能得到一个
正的奖励。我们当然希望我们的agent能够学会总是选择得到正奖励的arm。
"""

# list out our bandit arms
# currently arm 4 (index #3) is set to most often provide a positive reward
bandit_arms = [0.2, 0, -0.2, -2]
num_arms = len(bandit_arms)
def pullBandit(bandit):
	# get a random number
	result = np.random.randn(1)
	if result > bandit:
		# return a positive reward
		return 1
	else:
		# return a negative reward
		return -1


# the agent
"""
the code below established our simple neural agent. it consists of a set of values for each of the bandit arms. each value is an 
estimate of the value of the return from choosing the bandit. we use a policy gradient method to update the agent by moving the value 
for the selected action toward the recieved reward.
下面的代码展示了我们的神经元agent，它由每个bandit arm的一系列值组成。每一个值代表从选择bandit后返回的value的一个估计，我们用一个策略梯度方法来更新这个agent通过移动选择这个动作后得到的value趋于得到的奖励。

"""
tf.reset_default_graph()

# these two lines established the feed-forward part of the network
weights = tf.Variable(tf.ones([num_arms]))
output = tf.nn.softmax(weigths)

# the next six lines establish the training proceedure. we feed the reward and chosen action to the network to compute the loss, and use it to update the network
reward_holder = tf.placeholder(shape = [1], dtype = tf.float32)
action_holder = tf.placeholder(shape = [1], dtype = tf.int32)

responsible_output = tf.slice(output, action_holder, [1])
loss = -(tf.log(responsible_output) * reward_holder)
optimizer = tf.train.AdamOptimizer(learning_rate = 1e-3)
update = optimizer.minimize(loss)

# training agent

"""
we will train our agent by taking actions in our environment, and recieving rewards. using the rewards and actions, we can know how to 
properly update our network in order to more often choose actions that will yield the highest rewards over time.
我们通过在环境中选择动作，然后得到奖励，来训练agent。用奖励和动作，我们可以知道如何合适的更新我们的网络，从而随着时间的推移，让agent能够越来越能选择到能产生高奖励的动作。
"""
total_episodes = 1000 # set total number of episodes to train agent on
total_reward = np.zeros(num_arms) # set scoreboard for bandit arms to 0.

init = tf.global_variables_initializer()

# launch the tensorflow graph
with tf.Session() as sess:
	sess.run(init)
	i = 0
	while i < total_episodes:
		# choose action according to Boltzmann distribution
		actions = sess.run(output)
		a = np.random.choice(actions, p = actions) # 可以从一个int数字或1维array里随机选取内容，并将选取结果放入n维array中返回。
		action = np.argmax(actions == a)

		reward = pullBandit(bandit_arms[action]) # get our reward from picking one of the bandit arms

		# update the network
		_, resp, ww = sess.run([update, responsible_output, weights], feed_dict = {reward_holder: [reward], action_holder: [action]})

		# update our running tally of scores
		total_reward[action] += reward
		if i % 50 == 0:
			print("running reward for the " + str(num_arms) + " arms of the bandit: " + str(total_reward))
		i += 1
print( "\n the agent thinks arm " + str(np.argmax(ww) + 1) + " is the most promising... " )
if np.argmax(ww) == np.argmax(-np.array(bandit_arms)):
	print("...and it was right!")
else:
	print("...and it was wrong!")

	
"""
策略梯度算法与Q Learning的不同之处在于，它是直接输出动作，或者动作的概率，不需要输出Q值。

其它的基本流程不变：
1.选择一个动作action
2.得到奖励reward
3.根据动作action和奖励reward更新网络

最后奖励最大的就是我们要的结果。当然这个结果要在可行性范围内。

当然这是比较有代表性的policy gradience
"""
