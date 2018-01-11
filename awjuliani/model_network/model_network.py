# 3 model-based RL

# we implement a policy and model network which work in tandem to solve # the cartpole RL problem.

# loading libraries and staring CartPole environment
from __future__ import print_function
import numpy as np 
try:
	import cPickle as pickle 
except ModuleNotFoundError:
	import pickle
import tensorflow as tf 
import matplotlib.pyplot as plt 
import math

import sys
if sys.version_info.major > 2:
	xrange = range
del sys

import gym
env = gym.make('CartPole-v0')

# setting hyper-parameters
# hyperparameters
H = 8 # number of hidden layer neurons
learning_rate = 1e-2
gamma = 0.99 # discount factor for reward
decay_rate = 0.99 # decay factor for RMSProp leaky sum of grad 2
resume = False # resume from previous checkpoint?

model_bs = 3 # batch size when learning from model
real_bs = 3 # batch size when learning from real environment

# model initialization
D = 4 # input dimensionality

# policy network
tf.reset_default_graph()
observations = tf.placeholder(tf.float32, [None, 4], name = "input_x")
W1 = tf.get_variable("W1", shape = [4, H], initializer = tf.contrib.layers.xavier_initializer())
layer1 = tf.nn.relu(tf.matmul(observations, W1))
W2 = tf.get_variable("W2", shape = [H, 1], initializer = tf.contrib.layers.xavier_initializer())
score = tf.matmul(layer1, W2)
probability = tf.nn.sigmoid(score)

tvars = tf.trainable_variables()
input_y = tf.placeholder(tf.float32, [None, 1], name = "input_y")
advantages = tf.placeholder(tf.float32, name = "reward_signal")
adam = tf.train.AdamOptimizer(learning_rate = learning_rate)
W1Grad = tf.placeholder(tf.float32, name = "batch_grad1")
W2Grad = tf.placeholder(tf.float32, name = "batch_grad2")
batchGrad = [W1Grad, W2Grad]
loglik = tf.log(input_y * (input_y - probability) + (1 - input_y) * (input_y + probability))
loss = -tf.reduce_mean(loglik * advantages)
newGrads = tf.gradients(loss, tvars)
updateGrads = adam.apply_gradients(zip(batchGrad, tvars))
"""为什么有两个梯度？为什么要让梯度下降既作用于loss，又作用于vars变量？newGrad和updateGrad有什么关系，有什么用处？
策略梯度输入是状态，输出是动作的概率， 并且在梯度下降，也就是更新参数时，用评价指标advantage的期望，也就是与概率的对数似然的乘积。
log损失函数，也就是logistic逻辑回归经常用到的损失函数。"""

# model network
# here we implement a multi-layer neural network that predicts the next observation, reward, and done state from a current state and action.
"""
这里我们实现一个多层的神经网络，它预测下一个observation， reward， 并且根据当前的状态和动作得到新的状态。
"""
mH = 256 # model layer size

input_data = tf.placeholder(tf.float32, [None, 5])
with tf.variable_scope('rnnlm'):
	softmax_w = tf.get_variable("softmax_w", [mH, 50])
	softmax_b = tf.get_variable("softmax_b", [50])

previous_state = tf.placeholder( tf.float32, [None, 5], name = "previous_state")
W1M = tf.get_variable("W1M", shape = [5, mH], initializer = tf.contrib.layers.xavier_initializer())
B1M = tf.Variable(tf.zeros([mH]), name = "B1M")
layer1M = tf.nn.relu(tf.matmul(previous, W1M) + B1M)
W2M = tf.get_variable("W2M", shape = [mH, mH], initializer = tf.contrib.layers.xavier_initializer())
B2M = tf.Variable(tf.zeros([mH]), name = "B2M")
layer2M = tf.nn.relu( tf.matmul(layer1M, W2M) + B2M)
w0 = tf.get_variable("w0", shape = [mH, 4], initializer = tf.contrib.layers.xavier_initializer())
wR = tf.get_variable("wR", shape = [mH, 1], initializer = tf.contrib.layers.xavier_initializer())
wD = tf.get_variable("wD", shape = [mH, 1], initializer = tf.contrib.layers.xavier_initializer())

b0 = tf.Variable(tf.zeros([4]), name = "b0")
bR = tf.Variable(tf.zeros([1]), name = "bR")
bD = tf.Variable(tf.ones([1]), name = "bD")

predicted_observation = tf.matmul(layer2M, w0, name = "predicted_observation") + b0
predicted_reward = tf.matmul(layer2M, wR, name = "predicted_reward") + bR
predicted_done = tf.sigmoid(tf.matmul(layer2M, wD, name = "predicted_done") + bD)

true_observation = tf.placeholder(tf.float32, [None, 4], name = "true_observation")
true_reward = tf.placeholder(tf.float32, [None, 1], name = "true_reward")
true_done = tf.placeholder(tf.float32, [None, 1], name = "true_done")

predicted_state = tf.concat([predicted_observation, predicted_reward, predicted_done], 1)

observation_loss = tf.square(true_observation - predicted_observation)

reward_loss = tf.square(true_reward - predicted_reward)

done_loss = tf.multiply( predicted_done, true_done ) + tf.multiply(1 - predicted_done, 1 - true_done)
done_loss = -tf.log(done_loss)

model_loss = tf.reduce_mean(observation_loss + done_loss + reward_loss)

modelAdam = tf.train.AdamOptimizer(learning_rate = learning_rate)
updateModel = modelAdam.minimize(model_loss)


"""
输入previous state 经过一个隐含层神经网络得到结果，再各自经过一层分别得到预测的observation， reward， done，（把三者结合成predicted state）结合输入的真实的observation，reward， done得到对应各自的loss。
对observation， reward， done分别求loss，然后再组合成model loss
"""

# helper-functions
# 重置梯度buffer
def resetGradBuffer(gradBuffer):
	for ix, grad in enumerate(gradBuffer):
		gradBuffer[ix] = grad * 0
	return gradBuffer

# 计算打折的reward
def discount_rewards(r):
	"""take 1D float array of rewards and compute discounted reward"""
	discounted_r = np.zeros_like(r)
	running_add = 0
	for t in reversed(xrange(0, r.size)):
		running_add = running_add * gamma + r[t]
		discounted_r[t] = running_add
	return discounted_r

# this function uses our model to produce a new state when given a previous state and action
# 这个函数用我们的model，并根据上一个状态和动作产生一个新的状态。
def stepModel(sess, xs, action):
	toFeed = np.reshape(np.hstack([xs[-1][0], np.array(action)]), [1, 5])
	myPredict = sess.run([predicted_state], feed_dict = {previous_state: toFeed})
	reward = myPredict[0][:, 4]
	observation = myPredict[0][:, 0:4]
	observation[:, 0] = np.clip(observation[:, 0], -2.4, 2.4)
	observation[:, 2] = np.clip(observation[:, 2], -0.4, 0.4)
	doneP = np.clip(myPredict[0][:, 5], 0, 1)
	if doneP > 0.1 or len(xs) >= 300:
		done = True 
	else:
		done = False
	return observation, reward, done 

# training the policy and model

xs, drs, ys, ds = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 1
real_episodes = 1
init = tf.global_variables_initializer()
batch_size = real_bs

drawFromModel = False # when set to True, will use model for observations
trainTheModel = True # whether to train the model
trainThePolicy = False # whether to train the policy
switch_point = 1

# launch the graph
with tf.Session() as sess:
	rendering = False
	sess.run(init)
	observation = env.reset()
	x = observation
	gradBuffer = sess.run(tvars)
	gradBuffer = resetGradBuffer(gradBuffer)

	while episode_number <= 5000:
		# start displaying environment once performance is acceptably high
		if (reward_sum/batch_size > 150 and drawFromModel == False) or rendering == True:
			env.render()
			rendering = True

		x = np.reshape(observation, [1, 4])

		tfprob = sess.run(probability, feed_dict = {observations: x})
		action = 1 if np.random.uniform() < tfprob else 0

		# record various intermediates (needed later for backprop)
		xs.append(x)
		y = 1 if action == 0 else 0
		ys.append(y)

		# step the model or real environment and get new measurements
		if drawFromModel == False:
			observation, reward, done, info = env.step(action)
		else:
			observation, reward, done = stepModel(sess, xs, action)

		reward_sum += reward 

		ds.append( done * 1 )
		drs.append( reward ) # record reward (has to be done after we call step())

		if done:
			if drawFromModel == False:
				real_episodes += 1
			episode_number += 1

			# stack together all inputs, hidden states, action gradients, and rewards for this episode
			epx = np.vstack(xs)
			epy = np.vstack(ys)
			epr = np.vstack(drs)
			epd = np.vstack(ds)
			xs, drs, ys, ds = [], [], [], [] # reset array memory

			if trainTheModel == True:
				actions = np.array([np.abs(y-1) for y in epy][:-1])
				state_prevs = epx[:-1, :]
				state_prevs = np.hstack([state_prevs, actions])
				state_nexts = epx[1:, :]
				rewards = np.array(epr[1:, :])
				dones = np.array(epd[1:, :])
				state_nextsAll = np.hstack([state_nexts, rewards, dones])

				feed_dict = {previous_state: state_prevs, true_observation: state_nexts, true_done: dones, true_reward: rewards}
				loss, pState, _ = sess.run([model_loss, predicted_state, updateModel], feed_dict)

			if trainThePolicy == True:
				discounted_epr = discount_rewards(epr).astype('float32')
				discounted_epr -= np.mean(discounted_epr)
				discounted_epr /= np.std(discounted_epr)
				tGrad = sess.run(newGrads, feed_dict = {observations: epx, input_y: epy, advantages: discounted_epr})

				# if gradients becom too large, end training process
				if np.sum(tGrad[0] == tGrad[0]) == 0:
					break
				for ix, grad in enumerate(tGrad):
					gradBuffer[ix] += grad 

			if switch_point + batch_size == episode_number:
				switch_point = episode_number
				if trainThePolicy == True:
					sess.run( updateGrads, feed_dict = {W1Grad: gradBuffer[0], W2Grad: gradBuffer[1]})
					gradBuffer = resetGradBuffer(gradBuffer)

				running_reward = reward_sum if running_reward is None else running_reward * 0.99 + reward_sum * 0.01
				if drawFromModel == False:
					print('world perf: episode %f. reward %f. action: %f. mean reward %f.' % (real_episodes, reward_sum/real_bs, action, running_reward/real_bs))
					if reward_sum/batch_size > 200:
						break
				reward_sum = 0

				# once the model has been trained on 100 episodes, we start alternating between the policy from the model and training the model from the real environment.
				""" 一旦这个模型已经被训练了100轮，我就开始在。从模型中训练策略。和。从真实环境中训练模型。之间选择。"""
				if episode_number > 100:
					drawFromModel = not drawFromModel
					trainTheModel = not trainTheModel
					trainThePolicy = not trainThePolicy

			if drawFromModel == True:
				observation = np.random.uniform(-0.1, 0.1, [4]) # generate reasonable starting point
				batch_size = model_bs
			else:
				observation = env.reset()
				batch_size = real_bs

print(real_episodes)

# checking model representation
"""
here we can examine how well the model is able to approximate the true environment after training. the green line indicates the real environment, and the blue indicates model predictions
"""
plt.figure(figsize = (8, 12))
for i in range(6):
	plt.subplot(6, 2, 2 * i + 1)
	plt.plot(pState[:, i])
	plt.subplot(6, 2, 2*i + 1)
	plt.plot(state_nextsAll[:, i])
plt.tight_layout()  #图像外部边缘的调整可以使用plt.tight_layout()进行自动控制，此方法不能够很好的控制图像间的间隔。










"""
策略梯度算法是不是就是引入了advantage？reward 有递减的设计discount

本来是有环境的，在这里是说，可以用已经训练好的model，也可以继续训练model，当然还可以用policy network。

"""
