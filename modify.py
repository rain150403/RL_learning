"""
deep deterministic policy gradient (DDPG), reinforcement learning
DDPG is actor critic based algorithm
"""

import tensorflow as tf 
import numpy as np 
import gym

"""
np.random.seed()的作用：使得随机数据可预测。
当我们设置相同的seed，每次生成的随机数相同。如果不设置seed，则每次会生成不同的随机数
"""
np.random.seed(1)   
tf.set_random_seed(1)

#####################  hyper parameters  ####################

MAX_EPISODES = 200
MAX_EP_STEPS = 200
LR_A = 0.001    # learning rate for actor
LR_C = 0.001    # learning rate for critic
GAMMA = 0.9     # reward discount
REPLACEMENT = [
    dict(name='soft', tau=0.01),
    dict(name='hard', rep_iter_a=600, rep_iter_c=500)
][1]            # you can try different target replacement strategies
""" 你可以尝试不同的目标替换策略 """
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32

RENDER = False
OUTPUT_GRAPH = True
ENV_NAME = 'Pendulum-v0'


###############################  Actor  ####################################


class Actor(object):
	"""
	1.构建网络： actor网络，有两个网络 eval_net 和 target_net。每一个都是输入状态输出动作。 并把target_net输出的动作给critic。
	只有一个隐含层的网络， 输入状态， 输出动作，并将动作的取值限制在一定的范围内。输出SCALED_A
	2.学习: 进行训练， 批量更新。 到底什么是软替代和硬替代。
	3.选择动作： 输入单一状态，sess.run（action）得到要选择的动作 
	4.把梯度加入到graph中：策略梯度， y就是策略，也就是一个函数， x就是策略的参数，也就是自变量，求策略的梯度，也就是求函数的梯度，用于得到更多的Q值。（或者说是更好的Q值）
	这里的y就是动作self.a, x就是eval_net的参数self.e_params

	训练train， 梯度更新， train.AdamOptimizer， apply_gradients

	这里是演员，用策略梯度下降方法选择动作， 并输出给评论家。所以是输入状态，直接输出动作。
	"""
    def __init__(self, sess, action_dim, action_bound, learning_rate, replacement):
        self.sess = sess
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.lr = learning_rate
        self.replacement = replacement
        self.t_replace_counter = 0

        with tf.variable_scope('Actor'):
            # input s, output a
            self.a = self._build_net(S, scope='eval_net', trainable=True)

            # input s_, output a, get a_ for critic
            self.a_ = self._build_net(S_, scope='target_net', trainable=False)

        """收集变量"""
        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')

        """ 软替代（e要用权重分配法），硬替代（直接t， e代替），到底是在替代什么 """
        if self.replacement['name'] == 'hard':
            self.t_replace_counter = 0
            self.hard_replace = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
        else:
            self.soft_replace = [tf.assign(t, (1 - self.replacement['tau']) * t + self.replacement['tau'] * e)
                                 for t, e in zip(self.t_params, self.e_params)]

    def _build_net(self, s, scope, trainable):
        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.3)
            init_b = tf.constant_initializer(0.1)
            net = tf.layers.dense(s, 30, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l1',
                                  trainable=trainable)
            with tf.variable_scope('a'):
                actions = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, kernel_initializer=init_w,
                                          bias_initializer=init_b, name='a', trainable=trainable)
                scaled_a = tf.multiply(actions, self.action_bound, name='scaled_a')  # Scale output to -action_bound to action_bound
        return scaled_a

    def learn(self, s):   # batch update
        self.sess.run(self.train_op, feed_dict={S: s})

        if self.replacement['name'] == 'soft':
            self.sess.run(self.soft_replace)
        else:
            if self.t_replace_counter % self.replacement['rep_iter_a'] == 0:
                self.sess.run(self.hard_replace)
            self.t_replace_counter += 1

    def choose_action(self, s):
        s = s[np.newaxis, :]    # single state  增加维度  单一状态
        return self.sess.run(self.a, feed_dict={S: s})[0]  # single action  单一动作

    def add_grad_to_graph(self, a_grads):
        with tf.variable_scope('policy_grads'):
            # ys = policy;
            # xs = policy's parameters;
            # self.a_grads = the gradients of the policy to get more Q
            # tf.gradients will calculate dys/dxs with a initial gradients for ys, so this is dq/da * da/dparams
            self.policy_grads = tf.gradients(ys=self.a, xs=self.e_params, grad_ys=a_grads)

        with tf.variable_scope('A_train'):
            opt = tf.train.AdamOptimizer(-self.lr)  # (- learning rate) for ascent policy
            self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params))


###############################  Critic  ####################################

class Critic(object):
    def __init__(self, sess, state_dim, action_dim, learning_rate, gamma, replacement, a, a_):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.replacement = replacement

        with tf.variable_scope('Critic'):
        	""" 评论家网络，也是有两个网络eval_net, target_net。 
        	输入状态和动作， 输出Q值。  
        	应该说评论家是采用Q-learning方法，是输出Q值， 也就是说演员输出动作由评论家来评判。

        	eval_net：输入状态，动作（s, a）, 输出Q值： q(根据网络得出)
        	target_net: 输入状态，动作（s_, a_）, 输出Q值： q_(根据网络得出), 也就是为了q_target, 
        	target_q是根据基于演员actor网络的target_net产生的动作a_

        	target_q 是终极回报reward
        	TD_ERROR :是loss
        	c_train： 训练train， AdamOptimizer
        	a_grad：每个样本的梯度张量

        	这里也有一个软替代和硬替代的问题。


        	"""
            # Input (s, a), output q
            self.a = a
            self.q = self._build_net(S, self.a, 'eval_net', trainable=True)

            # Input (s_, a_), output q_ for q_target
            self.q_ = self._build_net(S_, a_, 'target_net', trainable=False)    # target_q is based on a_ from Actor's target_net

            self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval_net')
            self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target_net')

        with tf.variable_scope('target_q'):
            self.target_q = R + self.gamma * self.q_

        with tf.variable_scope('TD_error'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q))

        with tf.variable_scope('C_train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        with tf.variable_scope('a_grad'):
            self.a_grads = tf.gradients(self.q, a)[0]   # tensor of gradients of each sample (None, a_dim)

        if self.replacement['name'] == 'hard':
            self.t_replace_counter = 0
            self.hard_replacement = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)]
        else:
            self.soft_replacement = [tf.assign(t, (1 - self.replacement['tau']) * t + self.replacement['tau'] * e)
                                     for t, e in zip(self.t_params, self.e_params)]

    def _build_net(self, s, a, scope, trainable):
    	""" 建立网络，只有一个隐含层的网络， 输入状态和动作， 他们都各自有各自的权重， 结合得到net， 在经过最后一层全连接， 得到一个值，一个数字Q值。 Q（s, a）"""

        with tf.variable_scope(scope):
            init_w = tf.random_normal_initializer(0., 0.1)
            init_b = tf.constant_initializer(0.1)

            with tf.variable_scope('l1'):
                n_l1 = 30
                w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], initializer=init_w, trainable=trainable)
                w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], initializer=init_w, trainable=trainable)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=init_b, trainable=trainable)
                net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)

            with tf.variable_scope('q'):
                q = tf.layers.dense(net, 1, kernel_initializer=init_w, bias_initializer=init_b, trainable=trainable)   # Q(s,a)
        return q


    """ 学习，训练： sess.run(self.train_op, """

    def learn(self, s, a, r, s_):
        self.sess.run(self.train_op, feed_dict={S: s, self.a: a, R: r, S_: s_})
        if self.replacement['name'] == 'soft':
            self.sess.run(self.soft_replacement)
        else:
            if self.t_replace_counter % self.replacement['rep_iter_c'] == 0:
                self.sess.run(self.hard_replacement)
            self.t_replace_counter += 1


#####################  Memory  ####################

class Memory(object):
	""" 经历存储， 要不断地用新的经历取代旧的，还要采样用于训练 """
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.capacity  # replace the old memory with new memory
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n):
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]



env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
action_bound = env.action_space.high

# all placeholder for tf
with tf.name_scope('S'):
    S = tf.placeholder(tf.float32, shape=[None, state_dim], name='s')
with tf.name_scope('R'):
    R = tf.placeholder(tf.float32, [None, 1], name='r')
with tf.name_scope('S_'):
    S_ = tf.placeholder(tf.float32, shape=[None, state_dim], name='s_')


sess = tf.Session()

# Create actor and critic.
# They are actually connected to each other, details can be seen in tensorboard or in this picture:
""" 创建演员--评论家
它们实际上是互连的"""
actor = Actor(sess, action_dim, action_bound, LR_A, REPLACEMENT)
critic = Critic(sess, state_dim, action_dim, LR_C, GAMMA, REPLACEMENT, actor.a, actor.a_)
actor.add_grad_to_graph(critic.a_grads)

sess.run(tf.global_variables_initializer())

M = Memory(MEMORY_CAPACITY, dims=2 * state_dim + action_dim + 1)

if OUTPUT_GRAPH:
    tf.summary.FileWriter("logs/", sess.graph)

var = 3  # control exploration

for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0

    for j in range(MAX_EP_STEPS):

        if RENDER:
            env.render()

        # Add exploration noise
        a = actor.choose_action(s)
        a = np.clip(np.random.normal(a, var), -2, 2)    # add randomness to action selection for exploration
        s_, r, done, info = env.step(a)

        M.store_transition(s, a, r / 10, s_)

        if M.pointer > MEMORY_CAPACITY:
            var *= .9995    # decay the action randomness
            b_M = M.sample(BATCH_SIZE)
            b_s = b_M[:, :state_dim]
            b_a = b_M[:, state_dim: state_dim + action_dim]
            b_r = b_M[:, -state_dim - 1: -state_dim]
            b_s_ = b_M[:, -state_dim:]

            critic.learn(b_s, b_a, b_r, b_s_)
            actor.learn(b_s)

        s = s_
        ep_reward += r

        if j == MAX_EP_STEPS-1:
            print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
            if ep_reward > -300:
                RENDER = True
            break
            
            
           
 ###########################3   ddpg.py #######################################
 
 ##########################dppo.py ##########################33
 
 """
Proximal Policy Optimization (PPO)
Distributing workers in parallel to collect data, then stop worker's roll-out and train PPO on collected data.
Restart workers once PPO is updated.
The global PPO updating rule is adopted from DeepMind's paper (DPPO):
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym, threading, queue

EP_MAX = 1000
EP_LEN = 200
N_WORKER = 4                # parallel workers
GAMMA = 0.9                 # reward discount factor
A_LR = 0.0001               # learning rate for actor
C_LR = 0.0002               # learning rate for critic
MIN_BATCH_SIZE = 64         # minimum batch size for updating PPO
UPDATE_STEP = 10            # loop update operation n-steps
EPSILON = 0.2               # for clipping surrogate objective
GAME = 'Pendulum-v0'
S_DIM, A_DIM = 3, 1         # state and action dimension

""" 
跟A3C网络好像没太大区别， 有评论家critic和演员actor， 并且有一个全局网络，和多个worker，要多线程工作，只是这个更新过程没看明白。？？？"""
class PPO(object):
    def __init__(self):
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')

        # critic
        """
        ？？？
        这里涉及到value， advantage， 并且有discounted——reward，是policy gradient吗？
        """
        l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu)
        self.v = tf.layers.dense(l1, 1)
        self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
        self.advantage = self.tfdc_r - self.v
        self.closs = tf.reduce_mean(tf.square(self.advantage))
        self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        """
        这里是想说什么呢？
        无非就是用一个正态分布，选择动作而已， 按概率按分布选择。
        依然是输入状态，得到动作计算loss并，求梯度下降。
        """
        # actor
        pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
        self.sample_op = tf.squeeze(pi.sample(1), axis=0)  # operation of choosing action
        self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
        ratio = pi.prob(self.tfa) / (oldpi.prob(self.tfa) + 1e-5)
        surr = ratio * self.tfadv                       # surrogate loss

        self.aloss = -tf.reduce_mean(tf.minimum(        # clipped surrogate objective  修剪， 代理 对象
            surr,
            tf.clip_by_value(ratio, 1. - EPSILON, 1. + EPSILON) * self.tfadv))

        self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)
        self.sess.run(tf.global_variables_initializer())

    def update(self):
    	""" 这个更新过程要好好看看 """
        global GLOBAL_UPDATE_COUNTER
        while not COORD.should_stop():
            if GLOBAL_EP < EP_MAX:
                UPDATE_EVENT.wait()                     # wait until get batch of data
                self.sess.run(self.update_oldpi_op)     # copy pi to old pi
                data = [QUEUE.get() for _ in range(QUEUE.qsize())]      # collect data from all workers
                data = np.vstack(data)
                s, a, r = data[:, :S_DIM], data[:, S_DIM: S_DIM + A_DIM], data[:, -1:]
                adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
                # update actor and critic in a update loop
                [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(UPDATE_STEP)]
                [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(UPDATE_STEP)]
                UPDATE_EVENT.clear()        # updating finished
                GLOBAL_UPDATE_COUNTER = 0   # reset counter
                ROLLING_EVENT.set()         # set roll-out available

    def _build_anet(self, name, trainable):
    	""" 输入状态，经过一个隐含层， 再经过一个全连接层得到动作，X 1 = sigma， X 2 = mu， 产生一个正态分布， 并收集所有的参数"""
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfs, 200, tf.nn.relu, trainable=trainable)
            mu = 2 * tf.layers.dense(l1, A_DIM, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l1, A_DIM, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        return np.clip(a, -2, 2)


    """这里是获得discounted reward的函数"""
    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]

"""不执着于其中的道理，也不执着于最后的目的；我们不是要白首偕老，而是要相濡以沫， 我们相濡以沫， 不是执着于人生的苦痛，而是要放下心中的执着。"""
class Worker(object):
    def __init__(self, wid):
        self.wid = wid
        self.env = gym.make(GAME).unwrapped
        self.ppo = GLOBAL_PPO

    def work(self):
        global GLOBAL_EP, GLOBAL_RUNNING_R, GLOBAL_UPDATE_COUNTER
        while not COORD.should_stop():
            s = self.env.reset()
            ep_r = 0
            buffer_s, buffer_a, buffer_r = [], [], []
            for t in range(EP_LEN):
                if not ROLLING_EVENT.is_set():                  # while global PPO is updating
                    ROLLING_EVENT.wait()                        # wait until PPO is updated
                    buffer_s, buffer_a, buffer_r = [], [], []   # clear history buffer, use new policy to collect data
                a = self.ppo.choose_action(s)
                s_, r, done, _ = self.env.step(a)
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append((r + 8) / 8)                    # normalize reward, find to be useful
                s = s_
                ep_r += r

                GLOBAL_UPDATE_COUNTER += 1               # count to minimum batch size, no need to wait other workers
                if t == EP_LEN - 1 or GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
                    v_s_ = self.ppo.get_v(s_)
                    discounted_r = []                           # compute discounted reward
                    for r in buffer_r[::-1]:
                        v_s_ = r + GAMMA * v_s_
                        discounted_r.append(v_s_)
                    discounted_r.reverse()

                    bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                    buffer_s, buffer_a, buffer_r = [], [], []
                    QUEUE.put(np.hstack((bs, ba, br)))          # put data in the queue
                    if GLOBAL_UPDATE_COUNTER >= MIN_BATCH_SIZE:
                        ROLLING_EVENT.clear()       # stop collecting data
                        UPDATE_EVENT.set()          # globalPPO update

                    if GLOBAL_EP >= EP_MAX:         # stop training
                        COORD.request_stop()
                        break

            # record reward changes, plot later
            if len(GLOBAL_RUNNING_R) == 0: GLOBAL_RUNNING_R.append(ep_r)
            else: GLOBAL_RUNNING_R.append(GLOBAL_RUNNING_R[-1]*0.9+ep_r*0.1)
            GLOBAL_EP += 1
            print('{0:.1f}%'.format(GLOBAL_EP/EP_MAX*100), '|W%i' % self.wid,  '|Ep_r: %.2f' % ep_r,)


if __name__ == '__main__':
    GLOBAL_PPO = PPO()
    UPDATE_EVENT, ROLLING_EVENT = threading.Event(), threading.Event()
    UPDATE_EVENT.clear()            # not update now
    ROLLING_EVENT.set()             # start to roll out
    workers = [Worker(wid=i) for i in range(N_WORKER)]
    
    GLOBAL_UPDATE_COUNTER, GLOBAL_EP = 0, 0
    GLOBAL_RUNNING_R = []
    COORD = tf.train.Coordinator()
    QUEUE = queue.Queue()           # workers putting data in this queue
    threads = []
    for worker in workers:          # worker threads
        t = threading.Thread(target=worker.work, args=())
        t.start()                   # training
        threads.append(t)
    # add a PPO updating thread
    threads.append(threading.Thread(target=GLOBAL_PPO.update,))
    threads[-1].start()
    COORD.join(threads)

    # plot reward change and test
    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('Episode'); plt.ylabel('Moving reward'); plt.ion(); plt.show()
    env = gym.make('Pendulum-v0')
    while True:
        s = env.reset()
        for t in range(300):
            env.render()
            s = env.step(GLOBAL_PPO.choose_action(s))[0]


#################################################################
# simply DPPO

"""
A simple version of Proximal Policy Optimization (PPO) using single thread.

上面是多线程，下面是单线程。
这篇还是看看莫烦的文字解释吧
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym

EP_MAX = 1000
EP_LEN = 200
GAMMA = 0.9
A_LR = 0.0001
C_LR = 0.0002
BATCH = 32
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
S_DIM, A_DIM = 3, 1
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty
    dict(name='clip', epsilon=0.2),                 # Clipped surrogate objective, find this is better
][1]        # choose the method for optimization

"""难道我又发现新的优化方法了， 还是这种更新方法更适合概率分布问题？ KL penalty， 和 clipped surrogate objective（更好）"""

class PPO(object):

    def __init__(self):
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')

        # critic
        with tf.variable_scope('critic'):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu)
            self.v = tf.layers.dense(l1, 1)
            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            self.advantage = self.tfdc_r - self.v
            self.closs = tf.reduce_mean(tf.square(self.advantage))
            self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        # actor
        pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(pi.sample(1), axis=0)       # choosing action
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):
                # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
                ratio = pi.prob(self.tfa) / oldpi.prob(self.tfa)
                surr = ratio * self.tfadv
            if METHOD['name'] == 'kl_pen':
                self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                kl = tf.distributions.kl_divergence(oldpi, pi)
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))
            else:   # clipping method, find this is better
                self.aloss = -tf.reduce_mean(tf.minimum(
                    surr,
                    tf.clip_by_value(ratio, 1.-METHOD['epsilon'], 1.+METHOD['epsilon'])*self.tfadv))

        with tf.variable_scope('atrain'):
            self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)

        tf.summary.FileWriter("log/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def update(self, s, a, r):
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
        # adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful

        # update actor
        if METHOD['name'] == 'kl_pen':
            for _ in range(A_UPDATE_STEPS):
                _, kl = self.sess.run(
                    [self.atrain_op, self.kl_mean],
                    {self.tfs: s, self.tfa: a, self.tfadv: adv, self.tflam: METHOD['lam']})
                if kl > 4*METHOD['kl_target']:  # this in in google's paper
                    break
            if kl < METHOD['kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                METHOD['lam'] /= 2
            elif kl > METHOD['kl_target'] * 1.5:
                METHOD['lam'] *= 2
            METHOD['lam'] = np.clip(METHOD['lam'], 1e-4, 10)    # sometimes explode, this clipping is my solution
        else:   # clipping method, find this is better (OpenAI's paper)
            [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(A_UPDATE_STEPS)]

        # update critic
        [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(C_UPDATE_STEPS)]

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu, trainable=trainable)
            mu = 2 * tf.layers.dense(l1, A_DIM, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l1, A_DIM, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        return np.clip(a, -2, 2)

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]

env = gym.make('Pendulum-v0').unwrapped
ppo = PPO()
all_ep_r = []

for ep in range(EP_MAX):
    s = env.reset()
    buffer_s, buffer_a, buffer_r = [], [], []
    ep_r = 0
    for t in range(EP_LEN):    # in one episode
        env.render()
        a = ppo.choose_action(s)
        s_, r, done, _ = env.step(a)
        buffer_s.append(s)
        buffer_a.append(a)
        buffer_r.append((r+8)/8)    # normalize reward, find to be useful
        s = s_
        ep_r += r

        # update ppo
        if (t+1) % BATCH == 0 or t == EP_LEN-1:
            v_s_ = ppo.get_v(s_)
            discounted_r = []
            for r in buffer_r[::-1]:
                v_s_ = r + GAMMA * v_s_
                discounted_r.append(v_s_)
            discounted_r.reverse()

            bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
            buffer_s, buffer_a, buffer_r = [], [], []
            ppo.update(bs, ba, br)
    if ep == 0: all_ep_r.append(ep_r)
    else: all_ep_r.append(all_ep_r[-1]*0.9 + ep_r*0.1)
    print(
        'Ep: %i' % ep,
        "|Ep_r: %i" % ep_r,
        ("|Lam: %.4f" % METHOD['lam']) if METHOD['name'] == 'kl_pen' else '',
    )

plt.plot(np.arange(len(all_ep_r)), all_ep_r)
plt.xlabel('Episode');plt.ylabel('Moving averaged episode reward');plt.show()
