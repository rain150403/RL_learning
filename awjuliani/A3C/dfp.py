"""

A3C ： 网络是actor-critic网络，有好几个worker agent，分别在环境中自行学习，把经历给全局网络，
DFP ： 网络是Q network， 可能是dueling DQN，因为它在学习的时候是分别计算value和advantage，再结合成Q值。依然是有好几个worker，分别学习。
"""


# reinforcement learning with goals

# an implementation of learning to act by predicting the future
"""
怎么感觉可能除了输入，和上一节异步学习完全一样呢"""

import imageio
import multiprocessing
import threading
import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from helper import *
from gridworld_goals import *

# helper functions

# calculating temporal offsets(f)

"""
get_f takes a time-series of measurements as well as a set of temporal offsets, and produces the 'f' values for those measurements, which corresponds to how they change in the future at each offset.
get_f获取一个时间序列的测量以及一个时间差集合，并为这些测量值产生一个f值，这与在每一个偏移处未来如何改变有关。说白了就是这个函数给它一些自变量，得到一个因变量。

Given a set of offsets T, and a set of measurements in temporal order m, produces f as follows:
f = <m_T1  – m_0,m_T2  – m_0… m_Tn  – m_0>
"""
#experience buffer
""" experience buffer is used to store a history of experiences that can be randomly drawn from when training the network.
经历缓冲池， 用来存储历史经历，等到训练网络的时候可以随机采样。"""

def get_f(m,offsets):
    f = np.zeros([len(m),m.shape[1],len(offsets)])
    for i,offset in enumerate(offsets):
        f[:-offset,:,i] = m[offset:,:] - m[:-offset,:]
        if i > 0:
            f[-offset:,:,i] = f[-offset:,:,i-1]
    return f

class ExperienceBuffer():
    def __init__(self, buffer_size = 50000):
        self.buffer = []
        self.buffer_size = buffer_size
    
    def add(self,experience):
        if len(list(self.buffer)) + len(list(experience)) >= self.buffer_size:
            self.buffer[0:(len(list(experience))+len(list(self.buffer)))-self.buffer_size] = []
        self.buffer.extend(experience)
            
    def sample(self,size):
        return np.reshape(np.array(random.sample(self.buffer,size)),[size,5])


# direct future prediction (DFP) network
"""
This class contain the definition of the neural network in Tensorflow, including the tensorflow ops that will be required 
for updating the network. In the paper the authors refer to their model as "Direct Future Prediction (DFP)," and so I adopt the same notation.

就是定义了一个网络dfq，以及更新时需要用到的参数。
"""

class DFP_Network():
    def __init__(self,a_size,scope,trainer,num_offsets,num_measurements):
        with tf.variable_scope(scope):
            #Inputs and visual encoding layers
            self.observation = tf.placeholder(shape=[None,5,5,3],dtype=tf.float32)
            self.measurements = tf.placeholder(shape=[None,num_measurements],dtype=tf.float32)
            self.goals = tf.placeholder(shape=[None,num_measurements],dtype=tf.float32)
            self.hidden_o = slim.fully_connected(slim.flatten(self.observation),128,activation_fn=tf.nn.elu)
            self.hidden_m = slim.fully_connected(slim.flatten(self.measurements),64,activation_fn=tf.nn.elu)
            self.hidden_g = slim.fully_connected(slim.flatten(self.goals),64,activation_fn=tf.nn.elu)
            hidden_input = tf.concat([self.hidden_o,self.hidden_m,self.hidden_g],1)
            hidden_output = slim.fully_connected(hidden_input,256,activation_fn=tf.nn.elu)

            """
            要好好理解一下measurement和offset的意思才行。
            这里面是， 输入observation， measurement， goal， 三者先各自进入一个隐含层， 将得到的结果连接成为一个新的变量，再传入到下一个隐含层， 以上都是全连接。
            上述的到的结果再经过一次全连接分别计算expectation， 和advantage。然后再将两者结合。
            """

            #We calculate separate expectation and advantage streams, then combine then later
            #This technique is described in https://arxiv.org/pdf/1511.06581.pdf

            self.expectation = slim.fully_connected(hidden_output,a_size * num_offsets * num_measurements,
                activation_fn=None,
                biases_initializer=None)
            self.advantages = slim.fully_connected(hidden_output,a_size * num_offsets * num_measurements,
                activation_fn=None,
                biases_initializer=None)
            
            self.advantages = self.advantages - tf.reduce_mean(self.advantages,reduction_indices=1,keep_dims=True)
            self.prediction = self.expectation + self.advantages
            
            """ 再将得到的prediction也就是价值组合，reshape，得到自己想要的几个变量 [measurements X actions X offsets]"""
            #Reshape the predictions to be  [measurements x actions x offsets]
            self.prediction = tf.reshape(self.prediction, [-1,num_measurements,a_size,num_offsets])
            
            """下面就是动作选择策略了，要知道价值就直接对应动作，不是吗"""
            # We use a softmax with temperate to pick actions. This is instead of e-greedy.
            # For more info on action-selection strategies, see: 
            # goo.gl/oyL5Vx
            self.temperature = tf.placeholder(shape=[None],dtype=tf.float32)
            self.boltzmann = tf.nn.softmax(tf.reduce_sum(self.prediction,reduction_indices=3)/self.temperature)
            
            self.actions = tf.placeholder(shape=[None],dtype=tf.int32)
            self.actions_onehot = tf.one_hot(self.actions,a_size,dtype=tf.float32)
            
            # Select the predictions relevant to the chosen action.
            self.pred_action = tf.reduce_sum(self.prediction * tf.reshape(self.actions_onehot,[-1,1,a_size,1]), [2])
            
            """ 更新网络 """
            #Only the global network need ops for loss functions and gradient updating.
            if scope == 'global':
                self.target = tf.placeholder(shape=[None,num_measurements,num_offsets],dtype=tf.float32)
                
                #Loss function
                self.loss = tf.reduce_sum(tf.squared_difference(self.pred_action,self.target))
                
                #Sparsity of the action distribution
                self.entropy = -tf.reduce_sum(self.boltzmann * tf.log(self.boltzmann + 1e-7)) 

                #Get & apply gradients from network
                global_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'global')
                self.gradients = tf.gradients(self.loss,global_vars)
                self.var_norms = tf.global_norm(global_vars)
                grads,self.grad_norms = tf.clip_by_global_norm(self.gradients,9999.0)
                self.apply_grads = trainer.apply_gradients(list(zip(grads,global_vars)))


# Worker Agent
"""
With asynchronous learning we have multiple 'worker agents,' each of which interacts with their own environment and collects 
experiences using its own local network. At the end of an episode those experiences are sent to the experience buffer. A random 
batch of experiences are then drawn from the buffer and the 'global network' processes them and updates itself with backpropogation. 
The new global network is then copied over to the worker agent, and the process repeats.

和上一节异步学习是一样的。
有多个worker agents，每一个与它自己的环境进行交互，并用它自己的本地网络收集经历。每一轮的结束这些经历就被送到经历池中。随机的一批经历会被选中， 全局网络来处理它们，并用反向传播算法更新自己。
然后新的全局网络被拷贝到每一个worker agent。 然后整个过程重复。

"""
class Worker():
    def __init__(self,game,name,a_size,trainer,model_path,global_episodes,offsets,exp_buff,num_measurements,master,gif_path):
        self.name = "worker_" + str(name)
        self.number = name        
        self.offsets = offsets
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
        self.num_measurements = num_measurements
        self.summary_writer = tf.summary.FileWriter("train_"+str(self.number))

        """ 是在这里把参数更新到local network上的 """
        #Create the local copy of the network and the tensorflow op to copy global paramters to local network
        self.local_DFP = DFP_Network(a_size,self.name,trainer,len(offsets),num_measurements)
        self.update_local_ops = update_target_graph('global',self.name)        
        self.env = game
        
    def train(self,rollout,sess):
        rollout = np.array(rollout)
        measurements = np.vstack(rollout[:,2])
        targets = get_f(measurements,self.offsets) #Generate targets using measurements and offsets 为什么是用measurement和offset产生target呢？
        rollout[:,4] = list(zip(targets))
        self.exp_buff.add(list(zip(rollout)))  # 这也是要存到经历池的
        
        """获取一批经历用于更新全局神经网络,
        这里batch_size = 128， 传入的内容多了一些， observation，measurement， temperature， action， target， goals， ，更新之后得到的内容也增多了，loss， entropy， grad_norm,var_norm,apply_grads.只是这里的rollout是什么意思，为什么做分母。 """
        #Get a batch of experiences from the buffer and use them to update the global network
        if len(self.exp_buff.buffer) > 128:
            exp_batch = self.exp_buff.sample(128)
            feed_dict = {self.global_net.observation:np.stack(exp_batch[:,0],axis=0),
                self.global_net.measurements:np.vstack(exp_batch[:,2]),
                self.global_net.temperature:[0.1],
                self.global_net.actions:exp_batch[:,1],
                self.global_net.target:np.vstack(exp_batch[:,4]),
                self.global_net.goals:np.vstack(exp_batch[:,3])}
            loss,entropy,g_n,v_n,_ = sess.run([self.global_net.loss,
                self.global_net.entropy,
                self.global_net.grad_norms,
                self.global_net.var_norms,
                self.global_net.apply_grads],feed_dict=feed_dict)
            return loss / len(rollout), entropy / len(rollout), g_n,v_n
        else:
            return 0,0,0,0
        
    def work(self,sess,coord,saver,train):
        episode_count = sess.run(self.global_episodes)
        self.episode_count = episode_count
        total_steps = 0
        print("Starting worker " + str(self.number))
        with sess.as_default(), sess.graph.as_default():                 
            while not coord.should_stop():
                sess.run(self.update_local_ops) #Copy parameters from global to local network  训练之前当然要先把参数拷贝过来
                episode_buffer = []
                episode_frames = []
                d = False
                t = 0
                temp = 0.25 #How spread out we want our action distribution to be 我们希望我们的动作服从什么分布？
                
                s,o_big,m,g,h = self.env.reset()
                self.the_m = m

                while d == False:
                    """这里是我们的目标转换发生的地方。
                    如果电量低于 0.3 ， 我们的目标为优化电池
                    如果电量充足，我们的目标是优化无人机传递"""
                    #Here is where our goal-switching takes place 
                    # When the battery charge is below 0.3, we set the goal to optimize battery
                    # When the charge is above that value we set the goal to optimize deliveries
                    if m[1] <= .3:
                        self.g = np.array([0.0,1.0])
                    else:
                        self.g = np.array([1.0,0.0]) # 其实目标向量就是一个二值一维向量[0, 1]或者[1, 0]
                    a_dist = sess.run(self.local_DFP.boltzmann, 
                        feed_dict={
                        self.local_DFP.temperature:[temp],
                        self.local_DFP.observation:[s],
                        self.local_DFP.measurements:[m],
                        self.local_DFP.goals:[self.g]})
                    b = self.g*a_dist[0].T
                    c = np.sum(b,1)
                    c /= c.sum()
                    a = np.random.choice(c,p=c)
                    a = np.argmax(c == a)        
                    """ 以上很重要， 是通过神经网络之后，如何选择动作的很好的解释 .
                    要输入observation， measurement， goal， temperature才能得到结果"""
                    
                    s1,s1_big,m1,g1,h1,d = self.env.step(a)                        
                    episode_buffer.append([s,a,np.array(m),self.g,np.zeros(len(self.offsets))])
                    if self.name == 'worker_0' and episode_count % 150 == 0:
                        episode_frames.append(set_image_gridworld(s1_big,m1,t+1,g1,h1))
                    total_steps += 1
                    s = np.copy(s1)
                    m = []
                    m = m1[:]
                    g = g1[:]
                    h = h1
                    t += 1
                    
                    # End the episode after 100 steps  如果100步之后， 不论有没有达到目的，训练都会结束
                    if t > 100:
                        d = True
                                            
                self.episode_deliveries.append(m[0])
                self.episode_lengths.append(t)
                
                # Update the network using the experience buffer at the end of the episode.
                if train == True:
                    loss,entropy,g_n,v_n = self.train(episode_buffer,sess)
            
                    
                # Periodically save gifs of episodes, model parameters, and summary statistics.
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
                    summary.value.add(tag='Performance/Length', simple_value=float(mean_length))
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
num_measurements = 2 #Number of measurements
learning_rate = 1e-3 #Learning ragte
offsets = [1,2,4,8,16,32] # Set of temporal offsets
load_model = False #Whether to load a saved model
train = True #Whether to train the network
model_path = './model_goals' #Path to save the model to
gif_path = './frames_goals' #Path to save gifs of agent performance to

"""
The below code establishes the global tensorflow network, as well as creating and starting each of the workers with their own individual networks.
下面的代码展示了全局网络， 也创建并启动了每一个worker agent 各自的独立的网络。
"""
tf.reset_default_graph()

exp_buff = ExperienceBuffer() # 先收集经历

if not os.path.exists(model_path):
    os.makedirs(model_path)
    
if not os.path.exists(gif_path):
    os.makedirs(gif_path)

trainer = tf.train.AdamOptimizer(learning_rate=learning_rate)
master_network = DFP_Network(a_size,'global',trainer,len(offsets),num_measurements) # Generate global network
with tf.device("/cpu:0"): 
    global_episodes = tf.Variable(0,dtype=tf.int32,name='global_episodes',trainable=False)
    num_workers = multiprocessing.cpu_count() # Set workers ot number of available CPU threads
    workers = []
    # Create worker classes
    for i in range(num_workers):
        workers.append(
            Worker(gameEnv(partial=False,size=5),i,a_size,
            trainer,model_path,global_episodes,offsets,
            exp_buff,num_measurements,master_network,gif_path))
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
"""
1. 先收集经历
2. 实例化全局网络
3. 实例化worker network
4. worker开始work，开始工作，这里在每一轮就会涉及到train训练的过程，而train就会牵扯到global network

有多个worker agents，每一个与它自己的环境进行交互，并用它自己的本地网络收集经历。每一轮的结束这些经历就被送到经历池中。随机的一批经历会被选中， 
全局网络来处理它们，并用反向传播算法更新自己。
然后新的全局网络被拷贝到每一个worker agent。 然后整个过程重复。

虽然是这样说，但是我依然没找到全局网络在哪里出现过，尤其是它更新自己的地方
"""
