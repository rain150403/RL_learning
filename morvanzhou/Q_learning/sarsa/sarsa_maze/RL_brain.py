"""
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.
View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import numpy as np
import pandas as pd


class RL(object):
    def __init__(self, action_space, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = action_space  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy

        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index=self.q_table.columns,
                    name=state,
                )
            )

    def choose_action(self, observation):
        self.check_state_exist(observation)
        # action selection
        if np.random.rand() < self.epsilon:
            # choose best action
            state_action = self.q_table.ix[observation, :]
            state_action = state_action.reindex(np.random.permutation(state_action.index))     # some actions have same value
            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)
        return action

    def learn(self, *args):
        pass


# off-policy
class QLearningTable(RL):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(QLearningTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.ix[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.ix[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)  # update


# on-policy
class SarsaTable(RL):

    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        super(SarsaTable, self).__init__(actions, learning_rate, reward_decay, e_greedy)

    def learn(self, s, a, r, s_, a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.ix[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.ix[s_, a_]  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)  # update

        
        
"""

在莫烦的tutorials里面， 都是用pandas的dataframe生成Q-table的：
self.q_table = pd.DataFrame(columns = self.actions, dtype = np.float64)

这段代码，写了一个RL父类，这里实现了1）检验状态的存在 2）动作的选择，没有实现learn学习

这个learn函数是在子类里面实现的，因为对于sarsa和Q-learning，学习的过程是不一样的。

sarsa       \/    Q-table
Q-learning  /\   network

Q-learning 和sarsa的区别与联系：

Q-learning 是离线更新，所以在计算Q值时会选择最大的Q值的动作，所以说不用专门输入下一个状态所对应的动作。所以相比于sarsa输进learn函数的参数少一个动作
和sarsa的不同之处还在于
1.q-learning是在episode循环之内进行动作的选择，sarsa是在episode循环之外选择动作
2.q-learning是离线学习，sarsa是在线学习
3.q_learning勇敢，sarsa胆小
4.q_learning计算Q值时用最大值，sarsa更新Q值时，只考虑下一个动作。
或者
1.Q是offline，S是online learning
2.Q勇敢，S谨慎
3.S在每个episode开始选择一个动作，直到循环结束，Q每次循环都选择一个动作
4.Q每次都选Q值最大的动作，而S在一开始就“确定”了下一步该怎样走，也就是动作是确定的


这里并没有用神经网络，而是使用的Q-table。
但是更新过程还是 self.q_table.ix[s, a] += self.lr * (q_target - q_predict)

在线和离线学习的不同之处就在于，对于Q值的选择，离线学习，当然会选择Q值最大的那个，而在线学习就只能选择一个动作，下一时刻能产生的最大Q值的那个动作对应的Q值。

这里还涉及到继承的问题，有个超类RL 写了基本功能，只是没有实现learn，那么就在子类里分别实现各自的学习learn方法喽。
"""
