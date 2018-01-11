"""
policy-gradient solve the contextual bandit problem
上下文赌博机问题
"""

import tensorflow as tf 
import tensorflow.contrib.slim as slim
import numpy as np 

# the contextual bandits
"""
here we define our contextual bandits. in this example, we are using three four-armed bandit. 
what this means is that each bandit has four arms that canbe pulled. each bandit has different 
success probabilities for each arm, and as such requires different actions to obtain the best result. 
the pullBandit function generates a random number from a normal distribution with a mean of 0. the lower 
the bandit number , the more likely a positive reward will be returned. we want our agent to learn to 
always choose the bandit-arm that will most often give a positive reward, depending on the bandit presented.

这里我们定义上下文赌博机。在这个例子中，我们用三个四臂赌博机。也就是说每一个赌博机，都有四个arm可以被推动。每个
赌博机的每个arm都有不同的成功概率，因此需要不同的动作来获取最好的结果。这个函数从一个零均值正态分布产生一个随机数字。
赌博机的号码越小，就越有可能得到正的奖励。我们希望我们的agent能够学着总是在这个环境中选择能够给出正的奖励的arm。
"""
class contextual_bandit():
	def __init__(self):
		self.state = 0
		# list out our bandits. currently arms 4, 2, and 1 (respectively) are the most optimal.
		self.bandits = np.array([[0.2, 0, -0.0, -5], [0.1, -5, 1, 0.25], [-5, 5, 5, 5]])
		self.num_bandits = self.bandits.shape[0]
		self.num_actions = self.bandits.shape[1]

	def getBandit(self):
		self.state = np.random.randint(0, len(self.bandits)) # returns a random state for each episode.
		return self.state

	def pullArm(self, action):
		# get a random number
		bandit = self.bandits[self.state, action]
		result = np.random.randn(1)
		if result > bandit:
			# return a positive reward
			return 1
		else:
			# return a negative reward.
			return -1
# the policy-based agent
"""
the code below established our simple neural agent. it takes as input the current state, and returns an action.this allows the agent to 
take actions which are conditioned on the state of the environment, a critical step toward being able to solve full RL problems. the agent
uses a single set of weights, within which each value is an estimate of the value of the return from choosing a particular arm given a bandit.
we use a policy gradient method to update the agent by moving the value for the selected action toward the recieved reward.

下面的代码展示了我们简单的神经网络agent。 它输入当前状态，输出一个动作。这样允许agent依据当前的环境条件选择动作，一个重要的步骤使得有可能解决full RL 问题。
这个agent用一个单集合的weights，这里的每一个值是从环境中选择一个特定动作后得到的value的估计。我们用一个策略梯度方法来更新agent， 通过移动选择动作后得到的
value趋向于得到的奖励。
"""
class agent():
	def __init__(self, lr, s_size, a_size):
		# these lines established the feed-forward part of the network. the agent takes a state and produces an action.
		self.state_in = tf.placeholder(shape = [1], dtype = tf.int32)
		state_in_OH = slim.one_hot_encoding(self.state_in, s_size)
		output = slim.fully_connected(state_in_OH, a_size, biases_initializer = None, activation_fn = tf.nn.sigmoid, weights_initializer = tf.ones_initializer())
		self.output = tf.reshape( output, [-1] )
		self.chosen_action = tf.argmax(self.output, 0)

		# the next six lines establish the training proceedure. we feed the reward and chosen action into the network to compute the loss, and use it to update the network.
		self.reward_holder = tf.placeholder(shape = [1], dtype = tf.float32)
		self.action_holder = tf.placeholder(shape = [1], dtype = tf.int32)
		self.responsible_weight = tf.slice(self.output, self.action_holder, [1])
		self.loss = -(tf.log(self.responsible_weight) * self.reward_holder)
		optimizer = tf.train.GradientDescentOptimizer(learning_rate = lr)
		self.update = optimizer.minimize(self.loss)

"""
we will train our agent by getting a state from the environment, take an action , and recieve a reward. using these three things , we can know 
how to properly update our network in order to more often choose actions given states that will yield the highest rewards over time.

我们通过从环境中获取一个状态，采取一个动作，然后获得奖励来训练agent。用这三件事，我们可以知道如何合适的更新我们的网络。以便随着时间的推移让我们的agent能够越来越聪明，能够根据状态
选择到能得到更高奖励的动作。
"""
tf.reset_default_graph() # clear the tensorflow graph
cBandit = contextual_bandit() # load the bandits
myAgent = agent(lr = 0.001, s_size = cBandit.num_bandits, a_size = cBandit.num_actions) # load the agent
weights = trainable_variables()[0] # the weights we will evaluate to look into the network.

total_episodes = 10000 # set total number of episodes to train agent on.
total_reward = np.zeros([cBandit.num_bandits, cBandit.num_actions]) # set scoreboard for bandits to 0.
e = 0.1 # set the chance of taking a random action

init = tf.global_variables_initializer()

# launch the tensorflow graph
with tf.Session() as sess:
	sess.run(init)
	i = 0
	while i < total_episodes:
		s = cBandit.getBandit() # get a state from the environment

		# choose eigher a random action or one from our network
		if np.random.rand(1) < e:
			action = np.random.randint(cBandit.num_actions)
		else:
			action = sess.run(myAgent.chosen_action, feed_dict = {myAgent.state_in:[s]})

		reward = cBandit.pullArm(action) # get our reward for taking an action given a bandit.

		# update the network
		feed_dict = { myAgent.reward_holder: [reward], myAgent.action_holder: [action], myAgent.state_in: [s]}
		_, ww = sess.run([myAgent.update, weights], feed_dict = feed_dict)

		# update our running tally of scores
		total_reward[s, action] += reward 
		if i % 500 == 0:
			print( "mean reward for each of the " + str(cBandit.num_bandits) + " bandits: " + str(np.mean(total_reward, axis = 1)))
		i += 1
for a in range(cBandit.num_bandits):
	print("the agent thinks action " + str(np.argmax(ww[a]) + 1) + " for bandit " + str(a+1) + " is the most promising...")
	if np.argmax(ww[a]) == np.argmin(cBandit.bandits[a]):
		print("...and it was right!")
	else:
		print("...and it was wrong!")
