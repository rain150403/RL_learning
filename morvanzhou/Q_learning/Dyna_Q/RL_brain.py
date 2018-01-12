import numpy as np 
import pandas as pd 


"""
莫烦多用e_greedy的策略选择动作，就是不会陷入一个地方不出来，但是如果是直接选择，按理来说是可以直接得到最理想的结果的，总是选择最好的。这是肖师兄选择的方法。
"""
class QLearningTable:
	def __init__(self, actions, learning_rate = 0.01, reward_decay = 0.9, e_greedy = 0.9):
		self.actions = actions # a list
		self.lr = learning_rate
		self.gamma = reward_decay
		self.epsilon = e_greedy
		self.q_table = pd.DataFrame(columns = self.actions)

	def choose_action(self, observation):
		self.check_state_exist(observation)
		# action selection
		if np.random.uniform() < self.epsilon:
			# choose best action
			state_action = self.q_table.ix[observation, :]
			state_action = state_action.reindex(np.random.permutation(state_action.index))  # some actions have same value
			action = state_action.argmax()
		else:
			# choose random action
			action = np.random.choice(self.actions)
		return action

	def learn(self, s, a, r, s_):
		self.check_state_exist(s_)
		q_predict = self.q_table.ix[s, a]
		if s_ != 'terminal':
			q_target = r + self.gamma * self.q_table.ix[s_, :].max() # next state is not terminal
		else:
			q_target = r # next state is terminal
		self.q_table.ix[s, a] += self.lr * (q_target - q_predict) # update

	def check_state_exist(self, state):
		if state not in self.q_table.index:
			# append new state to q table
			self.q_table = self.q_table.append(
				pd.Series(
					[0] * len(self.actions), 
					index = self.q_table.columns, 
					name = state,
				)
			)

class EnvModel:
	"""
	这里单独又建了一个环境模型类，这里有怎么保存变换的信息，怎么采样经历，怎么返回奖励
	similar to the memory buffer in DQN, you can store past experiences in here.
	alternatively或者, the model can generate next state and reward signal accurately.
	"""
	def __init__(self, actions):
		# the simplest case is to think about the model is a memory which has all past transition information
		""" 最简单的情况就是把这个模型看成是一个memory，记录着之前的转换信息"""
		self.actions = actions
		self.database = pd.DataFrame(columns = actions, dtype = np.object)

	def store_transition(self, s, a, r, s_):
		if s not in self.database.index:
			self.database = self.databse.append(
				pd.Series(
					[None] * len(self.actions), 
					index = self.database.columns, 
					name = s,
				)
			)
		self.database.set_value(s, a, (r, s_))

	def sample_s_a(self):
		# 对状态和动作采样
		s = np.random.choice(self.database.index)
		a = np.random.choice(self.database.ix[s].dropna().index) # filter out the None value
		return s, a

	def get_r_s_(self, s, a):
		# 得到奖励
		r, s_ = self.database.ix[s, a]
		return r, s_ 
