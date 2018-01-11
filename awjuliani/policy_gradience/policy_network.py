# 2 policy-gradient method

# a policy-gradient based agent that can solve the CartPole problem

from __future__ import division

import numpy as np
try:
	import cPickle as pickle 
except:
	import pickle
import tensorflow as tf
# %matplotlib inline
import matplotlib.pyplot as plt 
import math

try:
	xrange = xrange
except:
	xrange = range

# 这里是一个很不错的用现成的环境做训练的例子。步骤很清晰
# loading the cartpole environment
import gym
env = gym.make('CartPole-v0')
# what happens if we try running the environment with random actions? how well do we do ? ( hint: not so well )
env.reset()
random_episodes = 0
reward_sum = 0
while random_episodes < 10:
	env.render()
	observation, reward, done, _ = env.step( np.random.randint(0, 2))
	reward_sum += reward
	if done:
		random_episodes += 1
		print( "reward for this episode was:", reward_sum )
		reward_sum = 0
		env.reset()

# 上面只是一个随机选择动作的例子，效果不是很好，下面我们看一个用policy神经网络的方法。
# setting up our neural network agent

"""
this time we will be using a policy neural network that takes observations，passes them through a single hidden layer, and then 
produces a probability of choosing a left/right movement.
这次我们会用一个policy neural network ， 它会将观测observations传入一个单隐含层，然后产生一个选择向左还是向右移动的概率。 
"""
# hyperparameters
H = 10 # number of hidden layer neurons
batch_size = 5 # every how many episodes to do a param update
learning_rate = 1e-2 # feel free to play with this to train faster or more stably
gamma = 0.99 # discount factor for reward

D = 4 # input dimensionality

tf.reset_default_graph()

# this defines the network as it goes from taking an observation of the environment to 
# giving a probability of chosing to the action of moving left or right.
"""这个网络的输入是从环境中得到的观测，输出是选择每个动作的概率"""
observations = tf.placeholder(tf.float32, [None, D], name = "input_x")
W1 = tf.get_variable("W1", shape = [D, H], initializer = tf.contrib.layers.xavier_initializer())
layer1 = tf.nn.relu(tf.matmul(observations, W1))
W2 = tf.get_variable("W2", shape = [H, 1], initializer = tf.contrib.layers.xavier_initializer())
score = tf.matmul(layer1, W2)
probability = tf.nn.sigmoid(score)

# from here we define the parts of the network needed for learning a good policy.
tvars = tf.trainable_variables()
input_y = tf.placeholder(tf.float32, [None, 1], name = "input_y")
advantages = tf.placeholder(tf.float32, name = "reward_signal")  # 没有shape吗？

# the loss function. this sends the weights in the direction of making actions that gave good 
# advantage (reward over time) more likely, and actions that didn't less likely. 
loglik = tf.log(input_y * (input_y - probability) + (1 - input_y) * ( input_y + probability ))
loss = -tf.reduce_mean(loglik * advantages)
newGrads = tf.gradients(loss, tvars)

# once we have collected a series of gradients from multiple episodes, we apply them.
# we don't just apply gradients after every episode in order to account for noise in the reward signal.
adam = tf.train.AdamOptimizer(learning_rate = learning_rate) # our optimizer
W1Grad = tf.placeholder(tf.float32, name = "batch_grad1") # placeholders to send the final gradients throungh when we update 
W2Grad = tf.placeholder(tf.float32, name = "batch_grad2")
batchGrad = [W1Grad, W2Grad]
updateGrads = adam.apply_gradients(zip(batchGrad, tvars))  
"""这里好好看看两个梯度是怎么办的"""


# advantage function
"""
this function allows us to weigh the rewards our agent recieves. in the context of the cartpole task, we want actions that kept the 
pole in the air a long time to have a large reward, and actions that contributed to the pole falling to have a decreased or negative
reward. we do this by weighing the rewards from the end of the episode, with actions at the end being seen as negative, since they 
likely contributed to the pole falling , and the episode ending. likewise, early actions are seen as more positive, since they weren't
responsible for the pole falling.

这个函数允许我们对agent得到的奖励做一个权衡。在cartpole任务的环境下，我们需要的动作是让这pole在空中停留的时间更长一些，以便得到更大的奖励，反之让它掉落的动作就会得到一个递减的或者负的奖励。
我们通过权衡在每一轮的结尾得到的奖励来选择动作，如果一个动作很有可能让pole掉落，那最终就会被视为负的，这一轮结束。同样的，一个动作不会让pole掉落，那我们就会认为它是正的。
"""
def discount_rewards(r):
	""" take 1D float array of rewards and compute discounted reward """
	discounted_r = np.zeros_like(r)
	running_add = 0
	for t in reserved(xrange(0, r.size)):
		running_add = running_add * gamma + r[t]
		discounted_r[t] = running_add
	return discounted_r

# running the agent and the environment
"""
here we run the neural nerwork agent, and have it act in the cartpole environment.
这里我们运行神经网络agent，让它运行在cartpole这个环境中。
"""

xs, hs, dlogps, drs, ys, tfps = [], [], [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 1
total_episodes = 10000
init = tf.global_variables_initializer()

# launch the graph
with tf.Session() as sess:
	rendering = False
	sess.run(init)
	observation = env.reset() # obtain an initial observation of the environment 

	# 梯度是怎么积累的
	# reset the gradient placeholder. we will collect gradients in gradBuffer until we are ready to update our policy network.
	gradBuffer = sess.run(tvars)
	for ix, grad in enumerate(gradBuffer):
		gradBuffer[ix] = grad * 0

	while episode_number <= total_episodes:
		# rendering the environment slows things down, so let's only look at it once our agent is doing a good job.
		if reward_sum / batch_size > 100 or rendering == True:
			env.render()
			rendering = True

		# make sure the observation is in a shape the network can handle.
		x = np.reshape(observation, [1, D])

		# run the policy network and get an action to take
		tfprob = sess.run(probability, feed_dict = {observations: x})
		action = 1 if np.random.uniform() < tfprob else 0

		xs.append(x) # observation
		y = 1 if action == 0 else 0 # a "fake label"
		ys.append(y)

		# step the environment and get new measurements
		observation, reward, done, info = env.step(action)
		reward_sum += reward

		drs.append(reward) # record reward (has to be done after we call step() to get reward for previous action)

		if done:
			episode_number += 1
			# stack together all inputs, hidden states, action gradients, and rewards for this episode
			epx = np.vstack(xs)
			epy = np.vstack(ys)
			epr = np.vstack(drs)
			tfp = tfps
			xs, hs, dlogps, drs, ys, tfps = [], [], [], [], [], [] # reset array memory

			# compute the discounted reward backwards through time
			discounted_epr = discount_rewards(epr)
			# size the rewards to be unit normal (helps control the gradient estimator variance)
			discounted_epr -= np.mean(discounted_epr)
			discounted_epr //= np.std(discounted_epr)

			# get the gradient for this episode, and save it in the gradBuffer
			tGrad = sess.run( newGrads, feed_dict = {observations: epx, input_y: epy, advantages: discounted_epr})
			for ix, grad in enumerate(tGrad):
				gradBuffer[ix] += grad
			
			# if we have completed enough episodes, then update the policy network with our gradients.
			if episode_number % batch_size == 0:
				sess.run(updateGrads, feed_dict = {W1Grad: gradBuffer[0], W2Grad: gradBuffer[1]})
				for ix, grad in enumerate(gradBuffer):
					gradBuffer[ix] = grad * 0

				# give a summary of how well our network is doing for each batch of episode.
				running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
				print('Average reward for episode %f. total average reward %f.' % (reward_sum//batch_size, running_reward//batch_size))

				if reward_sum//batch_size > 200:
					print("task solved in", episode_number, 'episodes!')
					break

				reward_sum = 0

			observation = env.reset()

print(episode_number, 'episodes completed')
# the network not only does much better than random actions, but achieves the goal of 200 points per episode, thus solving the task.
# 这个网络不仅比随机动作要好，而且每轮episode可以完成200点。（可能是想说快吧，产生的状态也多）

"""
advantages 在这里是什么呢？
"""
