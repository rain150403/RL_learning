# async-Q ,,, dfp

# reinforcement learning with Q-learning

import imageio
import multiprocessing
import threading
import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from helper import *
from gridworld_rewards import *


# helper functions
"""
ExperienceBuffer is used to store a history of experiences that can be randomly drawn from when training the network.
Additional helper functions are located in helper.py
ExperienceBuffer用于存储历史经历，以便训练的时候随机选择。
"""

class ExperienceBuffer():
    def __init__(self, buffer_size = 50000):
        self.buffer = []
        self.buffer_size = buffer_size
    
    def add(self,experience):
        if len(list(self.buffer)) + len(list(experience)) >= self.buffer_size:
            self.buffer[0:(len(list(experience))+len(list(self.buffer)))-self.buffer_size] = []
        self.buffer.extend(experience)
            
    def sample(self,size):
        return np.reshape(np.array(random.sample(self.buffer,size)),[size,3])

# Q-network
"""
This class contain the definition of the neural network in Tensorflow, including the tensorflow ops that will be required for updating the network.
这个类包含神经网络的定义，包括在网络更新时用到的参数ops。
"""
class QNetwork():
    def __init__(self,a_size,scope,trainer):
        with tf.variable_scope(scope):
            #Input and visual encoding layers
            self.observation = tf.placeholder(shape=[None,5,5,3],dtype=tf.float32)
            self.hidden_o = slim.fully_connected(slim.flatten(self.observation),128,activation_fn=tf.nn.elu)
            hidden_output = slim.fully_connected(self.hidden_o,256,activation_fn=tf.nn.elu)

            """ 输入observation， 经过两个隐含层之后， 再我们分别计算value和advantage，然后后面再结合。这里expectation = value """
            #We calculate separate value and advantage streams, then combine then later
            #This technique is described in https://arxiv.org/pdf/1511.06581.pdf
            self.expectation = slim.fully_connected(hidden_output,a_size,
                activation_fn=None,
                biases_initializer=None)
            self.advantages = slim.fully_connected(hidden_output,a_size,
                activation_fn=None,
                biases_initializer=None)
            self.advantages = self.advantages - tf.reduce_mean(self.advantages,reduction_indices=1,keep_dims=True)
            self.prediction = self.expectation + self.advantages
            """为什么要这样做呢？"""
            
            """一种新的动作选择方法"""
            # We use a softmax with temperate to pick actions. This is instead of e-greedy.
            # For more info on action-selection strategies, see: 
            # goo.gl/oyL5Vx
            
            self.temperature = tf.placeholder(shape=[None],dtype=tf.float32)
            self.boltzmann = tf.nn.softmax(self.prediction/self.temperature)
            
            self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
            self.actions_onehot = tf.one_hot(self.actions,a_size,dtype=tf.float32)
            
            self.pred_action = tf.reduce_sum(self.prediction * self.actions_onehot, [1])
            

            """难道只有global network才会有损失函数和梯度更新？"""
            #Only the global network need ops for loss functions and gradient updating.
            if scope == 'global':
                self.target = tf.placeholder(shape=[None],dtype=tf.float32)
                
                #Loss function
                self.loss = tf.reduce_sum(tf.squared_difference(self.pred_action,self.target))
                
                #Entropy tells us how diverse our action probabilities are
                self.entropy = -tf.reduce_sum(self.boltzmann * tf.log(self.boltzmann + 1e-7))

                #Get gradients from network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.gradients = tf.gradients(self.loss,global_vars)
                self.var_norms = tf.global_norm(global_vars)
                grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,9999.0)
                self.apply_grads = trainer.apply_gradients(zip(grads,global_vars))


# worker agent
"""
With asynchronous learning we have multiple 'worker agents,' each of which interacts with their own environment 
and collects experiences using its own local network. At the end of an episode those experiences are sent to the 
experience buffer. A random batch of experiences are then drawn from the buffer and the 'global network' processes 
them and updates itself with backpropogation. The new global network is then copied over to the worker agent, and the 
process repeats.

关于异步学习我们有多个worker agents， 每个agent和它自己的环境交互并用它自己的本地网络来收集经历。每一轮结束这些经历会被送进experience ExperienceBuffer。 
随机的一批经历会被选中global network会处理它们， 并用反向传播算法更新自己。新的全局网络global network会被拷贝给worker agent。然后整个过程重复。

"""
class Worker():
    def __init__(self,game,name,a_size,trainer,model_path,global_episodes,exp_buff,master,gif_path):
        self.name = "worker_" + str(name)
        self.number = name        
        self.global_net = master
        self.exp_buff = exp_buff
        self.model_path = model_path
        self.gif_path = gif_path
        self.trainer = trainer
        self.global_episodes = global_episodes
        self.increment = self.global_episodes.assign_add(1)
        self.episode_deliveries = []
        self.episode_lengths = []
        self.episode_mean_values = []
        self.summary_writer = tf.summary.FileWriter("train_"+str(self.number))

        #Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_Q = QNetwork(a_size,self.name,trainer)
        self.update_local_ops = update_target_graph('global',self.name)        
        self.env = game
    

    """先积累经历，直到经历数量大128， 采样128个经历用于训练。每个经历的第0维是observation， 第1维是action， 第2维是target，将这些feed进网络， 
    运行计算loss， grad_norms, var_norms， apply_grads，从而得到loss， grad，value相对应的值。"""
    def train(self,rollout,sess):
        rollout = np.array(rollout)  # rollout 首次展示
        self.exp_buff.add(zip(rollout))  # 这里应该是经历积累
        
        if len(self.exp_buff.buffer) > 128:
            exp_batch = self.exp_buff.sample(128)
            feed_dict = {self.global_net.observation:np.stack(exp_batch[:,0],axis=0),
                self.global_net.actions:exp_batch[:,1],
                self.global_net.target:exp_batch[:,2]}
            loss,g_n,v_n,_ = sess.run([self.global_net.loss,
                self.global_net.grad_norms,
                self.global_net.var_norms,
                self.global_net.apply_grads],feed_dict=feed_dict)
            return loss / len(rollout), g_n,v_n
        else:
            return 0,0,0
        
    def work(self,sess,coord,saver,train):
        episode_count = sess.run(self.global_episodes)
        self.episode_count = episode_count
        total_steps = 0
        print("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():                 
            while not coord.should_stop():
                sess.run(self.update_local_ops)
                episode_buffer = []
                episode_frames = []
                episode_rewards = 0
                d = False
                t = 0
                temp = 0.25
                
                s,o_big,m,g,h = self.env.reset()   # 也不知道这环境返回的是哪些值 return state,s_big,self.measurements,[self.goal.x,self.goal.y],[self.hero.x,self.hero.y]

                while d == False:
                    a_dist = sess.run(self.local_Q.boltzmann, 
                        feed_dict={self.local_Q.observation:[s],self.local_Q.temperature:[temp]})  # boltzmann有点像policy gradient里面的评价指标，这里看来是最终选择的动作
                    a = np.random.choice(a_dist[0],p=a_dist[0])
                    a = np.argmax(a_dist[0] == a)
                    
                    s1,s1_big,r,g1,h1,d = self.env.step(a)      # return state,s_big,reward,[self.goal.x,self.goal.y],[self.hero.x,self.hero.y],done
                    
                    #The Q-learning update rule
                    # We use this to generate target values to update our Q-network toward
                    """这里使用q-learning方法更新Q值， d = done，也就是说，如果已经结束，那就直接把r赋给y，否则，就要重新计算r值，再赋给y。"""
                    if d == True:
                        y = r
                    else:
                        self.qnext = sess.run(self.local_Q.prediction,feed_dict={self.local_Q.observation:[s1]})
                        y = r + 0.95*np.max(self.qnext)
                    
                    episode_rewards += r
                    episode_buffer.append([s,a,y])
                    if self.name == 'worker_0' and episode_count % 150 == 0:
                        episode_frames.append(set_image_gridworld_reward(s1_big,episode_rewards,t+1,g1,h1))
                    total_steps += 1
                    s = s1
                    g = g1
                    h = h1
                    t += 1
                    
                    if t > 100:
                        d = True
                                            
                self.episode_deliveries.append(episode_rewards)  # 这里是每一轮的reward积累的list
                self.episode_lengths.append(t) # 这里是记录每一轮运行了多少步的list
                
                # Update the network using the experience buffer at the end of the episode.
                if train == True:
                    loss,g_n,v_n = self.train(episode_buffer,sess)
                """其实worker的整个过程和原先是一样的，只不过训练更新网络的方法好像更简单，与全局网络有关。"""
            
                    
                # Periodically save gifs of episodes, model parameters, and summary statistics.
                """ 阶段性的保存每一轮的GIF图片，模型参数，以及总结的统计量 """
                if episode_count % 50 == 0 and episode_count != 0:
                    if episode_count % 2000 == 0 and self.name == 'worker_0' and train == True:
                        saver.save(sess,self.model_path+'/model-'+str(episode_count)+'.cptk')
                        print("Saved Model")

                    if self.name == 'worker_0' and episode_count % 150 == 0:
                        time_per_step = 0.25
                        self.images = np.array(episode_frames)
                        imageio.mimsave(self.gif_path+'/image'+str(episode_count)+'.gif',self.images, duration=time_per_step)
                        
                    mean_deliveries = np.mean(self.episode_deliveries[-50:])
                    mean_length = np.mean(self.episode_lengths[-50:])
                    mean_value = np.mean(self.episode_mean_values[-50:])
                    summary = tf.Summary()
                    summary.value.add(tag='Performance/Deliveries', simple_value=float(mean_deliveries))
                    if train == True:
                        summary.value.add(tag='Losses/Loss', simple_value=float(loss))
                        summary.value.add(tag='Losses/Grad Norm', simple_value=float(g_n))
                    self.summary_writer.add_summary(summary, episode_count)

                    self.summary_writer.flush()
                if self.name == 'worker_0':
                    sess.run(self.increment)
                episode_count += 1
                self.episode_count = episode_count


# training the network
# hyperparameters
a_size = 4 # Number of available actions
load_model = False #Whether to load the model or start training from scratch
train = True #Whether to train the model or simply use it for solving the task
model_path = './model_Q' #The location to save the model to
gif_path = './frames_Q' #The location to save gifs of the agent-environemnt interaction to


"""The below code establishes the global tensorflow network, as well as creating and starting each of the workers with their own individual networks.
下面的代码展示了全局网络， 也创建并启动了每一个worker agent各自独立的网络。
"""

tf.reset_default_graph()

exp_buff = ExperienceBuffer()

if not os.path.exists(model_path):
    os.makedirs(model_path)
    
if not os.path.exists(gif_path):
    os.makedirs(gif_path)

trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
master_network = QNetwork(a_size,'global',trainer) # Generate global network
with tf.device("/cpu:0"): 
    global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
    num_workers = multiprocessing.cpu_count() # Set workers ot number of available CPU threads
    workers = []
    # Create worker classes
    for i in range(num_workers):
        workers.append(
        Worker(gameEnv(partial=False,size=5),
        i,a_size,trainer,model_path,global_episodes,
        exp_buff,master_network,gif_path))
    saver = tf.train.Saver(max_to_keep=5)

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    if load_model == True:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(model_path)
        saver.restore(sess,ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())
        
    # Start each of the workers on a separate thread
    worker_threads = []
    for worker in workers:
        worker_work = lambda: worker.work(sess,coord,saver,train)
        thread = threading.Thread(target=(worker_work))
        thread.start()
        time.sleep(0.5)
        worker_threads.append(thread)
    coord.join(worker_threads)
