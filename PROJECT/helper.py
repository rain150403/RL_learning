# 这里讲述了项目中用到的三种强化学习方法

class Helper(object):
    def __init__(self, model, gamma):
        self.q_net = model    # 给定model，就是像ANN，CNN等经典网络结构，把这个网络结构赋值给q-net
        self.gamma = gamma

    def check(self, model):
        """
        check the data, for debugging
        :param model: the reference model
        :return: None
        """
        for p, rp in zip(self.q_net.parameters(), model.parameters()):
            assert torch.ge(p.data, rp.data).all()

    def get_y_target(self, batch):
        state, next_state, action, reward, terminal = batch

        next_state = Variable(next_state, volatile=True)  # 就是作用一下，让它变为变量
        q_value = self.q_net(next_state)   # 把下一个状态输入到q-net网络中，就能得到Q-value， Q值。
        target = Variable(reward)   # reward就是我们的target， 经过变量转换一下就好。
        max_q_value, _ = torch.max(q_value.data, dim=1)   # 找到最大的Q值
        mask = torch.ne(terminal, 1).float()   #  torch.ne(a, b) Implements != operator comparing each element in a with b (if b is a number) or each element in a with corresponding element in b.   有不同的就返回false吗？这个掩码是怎么用的呢？
        target.data.add_(self.gamma * mask * max_q_value)   # 这个公式不知道什么意思

        state = Variable(state)
        out = self.q_net(state)  # 输入状态到q-net网络，得到输出结果out
        y = torch.gather(out, dim=1, index=Variable(action).unsqueeze(1)).squeeze()  # 沿给定轴dim，将输入索引张量index指定位置的值进行聚合。都不知道这些是做什么的
        return y, target  # 为什么是返回两个值呢？
		# 从batch中得到状态、动作、奖励、终止等信息， 把next-state输入到网络q-net得到Q值，选出最大的Q值备用，算target比较麻烦了，首先是reward就是target，再根据mask、最大Q值、更新target
		#其次是把state输入Q-net得到out， 也是做了一个gather聚合， 得到y
		#可能就是当前状态之前的结果，与下一状态之前的结果，做一个时间差分？TD（error） 


class NatureHelper(Helper):
    def __init__(self, model, gamma):
        super(NatureHelper, self).__init__(model, gamma)
        self.target_net = copy.deepcopy(model)     # 从helper中继承而来，除了q-net这个网络，又引进了target-net， 只不过网络结构与q-net相同，也是ANN，或者CNN

    def get_y_target(self, batch):
        state, next_state, action, reward, terminal = batch   # 从batch中可以获得这么多信息：状态、动作、奖励、终止

        next_state = Variable(next_state)
        q_value = self.target_net(next_state)   # 在这里却是把下一个状态输入到target-net网络， 得到Q-值。
        max_q_value, _ = torch.max(q_value.data, dim=1)
        mask = torch.ne(terminal, 1).float()
        target = Variable(reward)
        target.data.add_(self.gamma * mask * max_q_value)

        state = Variable(state)
        out = self.q_net(state)
        y = torch.gather(out, dim=1, index=Variable(action).unsqueeze(1)).squeeze()
        return y, target
		"""
		总体来说，同样的是从batch中得到状态、动作等信息，然后把next-state输入到target-net网络， 得到Q-值。再选出Q值最大的，备用。同样是利用reward、mask、最大Q值等计算target。
		把state输入到q-net得到out，然后再做了一个gather聚合，得到y
		
		也就是这里依然是返回两个值y，target。
		
		
		也就是说上一个方法， state、next-state都输入到q-net， 而这里next-state输入target-net， state输入到q-net，其它是一样的。
		"""
    def update(self):
        self.target_net.load_state_dict(self.q_net.state_dict())
		# 这里是状态的更新

class DoubleHelper(Helper):
    def __init__(self, model, gamma):
        super(DoubleHelper, self).__init__(model, gamma)
        self.q_net2 = copy.deepcopy(model)  # 从helper中继承而来，除了q-net这个网络，又引进了q-net2而不是target-net。

    def get_y_target(self, batch):
        state, next_state, action, reward, terminal = batch
        target = Variable(reward)
        mask = torch.ne(terminal, 1).float()
        state = Variable(state)
        next_state = Variable(next_state)

        # we use the first net to choose action, the second to evaluate  用第一个网络选择动作，用第二个网络评估， 也就是actor-critic方法
        q_value = self.q_net(next_state)
        _, max_action = torch.max(q_value.data, dim=1, keepdim=True)
        # the second to evaluate
        q_value2 = self.q_net2(next_state)
        q_value_evaluate = torch.gather(q_value2.data, dim=1, index=max_action).squeeze()
        target.data.add_(self.gamma * mask * q_value_evaluate)
        out = self.q_net(state)
        y = torch.gather(out, dim=1, index=Variable(action).unsqueeze(1)).squeeze()
		# 依然是先输入next-state， 只不过这里是把next-state输入到两个网络， q-net得到q_value， 用于选择max_action， 最优动作， q-net2得到q_value2用于计算q_value_evaluate， 并用这个估计Q值去计算target， 
		
		# state输入到q-net得到out， 聚合成y；   由此，就可以得到我们的两个值，y， target
        return y, target

    def update(self):
        self.q_net2.load_state_dict(self.q_net.state_dict()) # 依然是更新状态。
