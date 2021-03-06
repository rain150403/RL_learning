# 4.deep Q-networks and beyond

# i implement a deep Q-Network using both double DQN and dueling DQN.
# the agent learn to solve a navigation task in a basic grid world.

from __future__ import division

import gym
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import scipy.misc
import os

# Load the game environment
"""
Feel free to adjust the size of the gridworld. Making it smaller provides an easier task for our DQN agent, while making 
the world larger increases the challenge.

你可以随意调整grid world的大小，当然越小就越简单，越大也也困难。

The position of the three blocks is randomized every episode.
还有，每一轮陷阱和目标块的位置， 还有agent的初始位置都是随机的。
"""

from gridworld import gameEnv

env = gameEnv(partial = False, size = 5)

# implementing the network itself

class Qnetwork():
    def __init__(self,h_size):
        #The network recieves a frame from the game, flattened into an array.
        #It then resizes it and processes it through four convolutional layers.
        """获取游戏的一帧，拉平成为一个数组，再resize，然后用4个卷积层去处理，用卷积层处理，能更好的“理解”图像的信息"""
        self.scalarInput =  tf.placeholder(shape=[None,21168],dtype=tf.float32)
        self.imageIn = tf.reshape(self.scalarInput,shape=[-1,84,84,3])
        self.conv1 = slim.conv2d( \
            inputs=self.imageIn,num_outputs=32,kernel_size=[8,8],stride=[4,4],padding='VALID', biases_initializer=None)
        self.conv2 = slim.conv2d( \
            inputs=self.conv1,num_outputs=64,kernel_size=[4,4],stride=[2,2],padding='VALID', biases_initializer=None)
        self.conv3 = slim.conv2d( \
            inputs=self.conv2,num_outputs=64,kernel_size=[3,3],stride=[1,1],padding='VALID', biases_initializer=None)
        self.conv4 = slim.conv2d( \
            inputs=self.conv3,num_outputs=h_size,kernel_size=[7,7],stride=[1,1],padding='VALID', biases_initializer=None)
        
        #We take the output from the final convolutional layer and split it into separate advantage and value streams.
        """经过四个卷积的处理之后，我们把结果分为advantage和value"""
        self.streamAC,self.streamVC = tf.split(self.conv4,2,3)
        self.streamA = slim.flatten(self.streamAC)
        self.streamV = slim.flatten(self.streamVC)
        xavier_init = tf.contrib.layers.xavier_initializer()
        self.AW = tf.Variable(xavier_init([h_size//2,env.actions]))
        self.VW = tf.Variable(xavier_init([h_size//2,1]))
        self.Advantage = tf.matmul(self.streamA,self.AW)
        self.Value = tf.matmul(self.streamV,self.VW)
        
        #Then combine them together to get our final Q-values.
        """再把advantage和value组合在一起成为我们的最终的Q-value"""
        self.Qout = self.Value + tf.subtract(self.Advantage,tf.reduce_mean(self.Advantage,axis=1,keep_dims=True))
        self.predict = tf.argmax(self.Qout,1)
        
        #Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        """利用Q-value的目标值和预测值的最小二乘损失获得loss"""
        self.targetQ = tf.placeholder(shape=[None],dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions,env.actions,dtype=tf.float32)
        
        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)
        
        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)



# experience replay
""" This class allows us to store experies and sample then randomly to train the network.
这个类允许我们存储经历，然后随机采样用于训练网络。
"""
class experience_buffer():
	def __init__(self, buffer_size = 50000):
		self.buffer = []
		self.buffer_size = buffer_size

	def add(self, experience):
		if len(self.buffer) + len(experience) >= self.buffer_size:
			self.buffer[0: (len(experience) + len(self.buffer)) - self.buffer_size] = []
		self.buffer.extend(experience)

	def sample(self, size):
		return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])

""" this is a simple function to resize our game frames. """
def processState(states):
	return np.reshape(states, [21168])

""" these functions allow us to update the parameters of our target network with those of the primary network
这个函数帮助我们更新新网络训练的参数到目标网络。

但是我确实没看懂是怎么更新参数的？？？
"""

def updateTargetGraph(tfVars, tau):
	total_vars = len(tfVars)
	op_holder = []
	for idx, var in enumerate(tfVars[0: total_vars//2]):
		op_holder.append(tfVars[idx + total_vars//2].assign((var.value()* tau) + ((1-tau) * tfVars[idx + total_vars//2].value())))
	return op_holder
def updateTarget(op_holder, sess):
	for op in op_holder:
		sess.run(op)


# training the network

# setting all the training parameters

batch_size = 32 #How many experiences to use for each training step.
update_freq = 4 #How often to perform a training step.
y = .99 #Discount factor on the target Q-values
startE = 1 #Starting chance of random action
endE = 0.1 #Final chance of random action
annealing_steps = 10000. #How many steps of training to reduce startE to endE.
num_episodes = 10000 #How many episodes of game environment to train network with.
pre_train_steps = 10000 #How many steps of random actions before training begins.
max_epLength = 50 #The max allowed length of our episode.
load_model = False #Whether to load a saved model.
path = "./dqn" #The path to save our model to.
h_size = 512 #The size of the final convolutional layer before splitting it into Advantage and Value streams.
tau = 0.001 #Rate to update target network toward primary network

tf.reset_default_graph()
mainQN = Qnetwork(h_size)
targetQN = Qnetwork(h_size)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

trainables = tf.trainable_variables()

targetOps = updateTargetGraph(trainables,tau)

myBuffer = experience_buffer()

#Set the rate of random action decrease. 
e = startE
stepDrop = (startE - endE)/annealing_steps

#create lists to contain total rewards and steps per episode
jList = []
rList = []
total_steps = 0

#Make a path for our model to be saved in.
if not os.path.exists(path):
    os.makedirs(path)

with tf.Session() as sess:
    sess.run(init)
    if load_model == True:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess,ckpt.model_checkpoint_path)
    for i in range(num_episodes):
        episodeBuffer = experience_buffer()
        #Reset environment and get first new observation
        s = env.reset()
        s = processState(s)
        d = False
        rAll = 0
        j = 0
        #The Q-Network
        while j < max_epLength: #If the agent takes longer than 200 moves to reach either of the blocks, end the trial.
            j+=1
            #Choose an action by greedily (with e chance of random action) from the Q-network
            if np.random.rand(1) < e or total_steps < pre_train_steps:
                a = np.random.randint(0,4)
            else:
                a = sess.run(mainQN.predict,feed_dict={mainQN.scalarInput:[s]})[0]
            s1,r,d = env.step(a)
            s1 = processState(s1)
            total_steps += 1
            episodeBuffer.add(np.reshape(np.array([s,a,r,s1,d]),[1,5])) #Save the experience to our episode buffer.
            
            if total_steps > pre_train_steps:
                if e > endE:
                    e -= stepDrop
                
                if total_steps % (update_freq) == 0:
                    trainBatch = myBuffer.sample(batch_size) #Get a random batch of experiences.
                    #Below we perform the Double-DQN update to the target Q-values
                    Q1 = sess.run(mainQN.predict,feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,3])})
                    Q2 = sess.run(targetQN.Qout,feed_dict={targetQN.scalarInput:np.vstack(trainBatch[:,3])})
                    end_multiplier = -(trainBatch[:,4] - 1)
                    doubleQ = Q2[range(batch_size),Q1]
                    targetQ = trainBatch[:,2] + (y*doubleQ * end_multiplier)
                    #Update the network with our target values.
                    _ = sess.run(mainQN.updateModel, \
                        feed_dict={mainQN.scalarInput:np.vstack(trainBatch[:,0]),mainQN.targetQ:targetQ, mainQN.actions:trainBatch[:,1]})
                    
                    updateTarget(targetOps,sess) #Update the target network toward the primary network.
            rAll += r
            s = s1
            
            if d == True:

                break
        
        myBuffer.add(episodeBuffer.buffer)
        jList.append(j)
        rList.append(rAll)
        #Periodically save the model. 
        if i % 1000 == 0:
            saver.save(sess,path+'/model-'+str(i)+'.ckpt')
            print("Saved Model")
        if len(rList) % 10 == 0:
            print(total_steps,np.mean(rList[-10:]), e)
    saver.save(sess,path+'/model-'+str(i)+'.ckpt')
print("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")
"""参变量太多，看的有点晕
没看明白是怎么训练更新，并更新参数到target网络的，
也没看明白double DQN 更没看到dueling DQN"""

# checking network learning

# mean reward over time

rMat = np.resize(np.array(rList), [len(rList)//100, 100])
rMean = np.average(rMat, 1)
plt.plot(rMean)

"""
Q network网络结构：

#The network recieves a frame from the game, flattened into an array.
#It then resizes it and processes it through four convolutional layers.
获取游戏的一帧，拉平成为一个数组，再resize，然后用4个卷积层去处理，用卷积层处理，能更好的“理解”图像的信息

#We take the output from the final convolutional layer and split it into separate advantage and value streams.
经过四个卷积的处理之后，我们把结果分为advantage和value

#Then combine them together to get our final Q-values.
再把advantage和value组合在一起成为我们的最终的Q-value

#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
利用Q-value的目标值和预测值的最小二乘损失获得loss

参数：
每一个训练步需要多少经历（每次从experience buffer获取样本数）： batch_size
多久执行一次训练步（每隔多少step执行一次模型参数更新）： update_freq   更新频率
游戏训练多少轮： num_episodes
在训练开始之前执行多少次动作选择（目的是经历experience积累）： pre_train_steps
学习率（在更新参数的时候涉及到的学习率）： tau
每一轮最多执行的步数，一旦超过这个数字，还没有结果，那就自动结束： max_epLength
startE起始执行随机Action概率。
endE最终执行随机Action概率。
anneling_steps从初始随机概率降到最终随机概率所需步数。

（这就是前面提到的，随着步数的增加，选择随机动作的概率越来越小。）





mainQN,target QN这两个网络是用来干嘛的，有什么关系？

没看明白是怎么训练更新，并更新参数到target网络的，

在这里找double DQN, dueling DQN的痕迹。

https://my.oschina.net/u/3482787/blog/1506974这个写的挺详细的。
"""
