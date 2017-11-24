'''
待删减
'''
from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import os
import shutil
import numpy as np
import random
from itertools import count
import copy

from util import *
import time

use_tensorboard = True
try:
    from tensorboard import SummaryWriter
except:
    use_tensorboard = False
    print('Dont use tensorboard')

use_cuda = torch.cuda.is_available()


def set_use_cuda(enable):
    """
    Enable CUDA or not. Modify the global variable `use_cuda`.
    :param enable: enable CUDA
    :return: None
    """
    global use_cuda
    use_cuda = enable


def load_checkpoint(model_name):
    """
    Load pre-trained checkpoint from disk.
    :param model_name: the model file name
    :return: the `state_dict` of weights and other info as `dict`
    """
    try:
        checkpoint = torch.load(model_name)
    except:
        checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
    state_dict = checkpoint['state_dict']
    info = {}
    for k, v in checkpoint.items():
        if k == 'state_dict':
            continue
        info[k] = v
    return state_dict, info


def get_batch(replay_buffer, batch_size, use_prioritized, beta):
    """
    Get batch using randomly sampling from memory for experience replay.
    :param memory: the memory storing `(st, st+1, a, r, t)`
    :param batch_size: batch size
    :return: samples
    """
    if use_prioritized:
        batch = replay_buffer.sample(batch_size, beta)
        state, next_state, action, reward, terminal, weights, batch_idx = batch

    else:
        batch = replay_buffer.sample(batch_size)
        state, next_state, action, reward, terminal = batch

    state = torch.from_numpy(state).float()
    next_state = torch.from_numpy(next_state).float()
    action = torch.from_numpy(action)
    reward = torch.from_numpy(reward).float()
    terminal = torch.from_numpy(terminal).long()

    if use_cuda:
        state, next_state, action, reward, terminal = \
        state.cuda(), next_state.cuda(), action.cuda(),\
        reward.cuda(), terminal.cuda()

    if not use_prioritized:
        return state, next_state, action, reward, terminal
    else:
        weights = torch.from_numpy(weights).float()
        if use_cuda:
            weights = weights.cuda()

        return state, next_state, action, reward, terminal, weights, batch_idx

def get_batch_ori(mem, bsz):
    batch = random.sample(mem, bsz)
    state = np.array([b[0] for b in batch], dtype=np.float32)
    next_state = np.array([b[1] for b in batch], dtype=np.float32)
    action = np.array([b[2] for b in batch])
    reward = np.array([b[3] for b in batch], dtype=np.float32)
    ter = np.array([b[4] for b in batch])

    state = torch.from_numpy(state)
    next_state = torch.from_numpy(next_state)
    action = torch.from_numpy(action)
    reward = torch.from_numpy(reward)
    ter = torch.from_numpy(ter)

    if use_cuda:
        state, next_state, action, reward, ter = \
          state.cuda(), next_state.cuda(), action.cuda(), \
          reward.cuda(), ter.cuda()

    return state, next_state, action, reward, ter


def get_action_optimally(model, state):
    """
    Get optimal action given current state
    :param model: model to estimate q value
    :param state: current state
    :return: action index
    """
    state = state.astype(np.float32)
    var = Variable(torch.from_numpy(state), volatile=True).unsqueeze(0)
    if use_cuda:
        var = var.cuda()
    q_value = model(var)
    _, idx = torch.max(q_value.data, dim=1)
    idx = idx.squeeze()
    return idx[0], q_value.data.squeeze()

def get_action_with_probability(model, state):
    """
    Get the action w.r.t probability(normalized q value). Not TEST.
    :param model:
    :param state:
    :return:
    """
    var = Variable(torch.from_numpy(state), volatile=True).unsqueeze(0)
    if use_cuda:
        var = var.cuda()
    q_value = model(var)
    idx = torch.multinomial(q_value.data.squeeze(), 1)
    return idx[0]


class Stat:
    def __init__(self):
        self.notes = {}
        self.time_step = 0

    def regester_key(self, key, init_val):
        self.notes[key] = init_val

    def update_value(self, key, val):
        self.notes[key] += val

    def update(self, **kwargs):
        self.time_step += 1
        for k, v in kwargs.items():
            if k in self.notes.keys():
                self.notes[k] += v
            else:
                self.notes[k] = v

    def reset(self):
        self.time_step = 0
        for k in self.notes.keys():
            self.notes[k] = 0

    def get_time_step(self):
        return self.time_step

    def get_value(self, key):
        if key not in self.notes.keys():
            print('Not find the key: {}'.format(key))
            print('Available keys: {}'.format(list(self.notes.keys())))
        return self.notes[key]

    def get_average(self, key):
        if key not in self.notes.keys():
            print('Not find the key: {}'.format(key))
            print('Available keys: {}'.format(list(self.notes.keys())))
            raise KeyError
        return float(self.notes[key])/self.time_step


class Helper(object):
    def __init__(self, model, gamma):
        self.q_net = model
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

        next_state = Variable(next_state, volatile=True)
        q_value = self.q_net(next_state)
        target = Variable(reward)
        max_q_value, _ = torch.max(q_value.data, dim=1)
        mask = torch.ne(terminal, 1).float()
        target.data.add_(self.gamma * mask * max_q_value)

        state = Variable(state)
        out = self.q_net(state)
        y = torch.gather(out, dim=1, index=Variable(action).unsqueeze(1)).squeeze()
        return y, target


class NatureHelper(Helper):
    def __init__(self, model, gamma):
        super(NatureHelper, self).__init__(model, gamma)
        self.target_net = copy.deepcopy(model)

    def get_y_target(self, batch):
        state, next_state, action, reward, terminal = batch

        next_state = Variable(next_state)
        q_value = self.target_net(next_state)
        max_q_value, _ = torch.max(q_value.data, dim=1)
        mask = torch.ne(terminal, 1).float()
        target = Variable(reward)
        target.data.add_(self.gamma * mask * max_q_value)

        state = Variable(state)
        out = self.q_net(state)
        y = torch.gather(out, dim=1, index=Variable(action).unsqueeze(1)).squeeze()
        return y, target

    def update(self):
        self.target_net.load_state_dict(self.q_net.state_dict())


class DoubleHelper(Helper):
    def __init__(self, model, gamma):
        super(DoubleHelper, self).__init__(model, gamma)
        self.q_net2 = copy.deepcopy(model)

    def get_y_target(self, batch):
        state, next_state, action, reward, terminal = batch
        target = Variable(reward)
        mask = torch.ne(terminal, 1).float()
        state = Variable(state)
        next_state = Variable(next_state)

        # we use the first net to choose action, the second to evaluate
        q_value = self.q_net(next_state)
        _, max_action = torch.max(q_value.data, dim=1, keepdim=True)
        # the second to evaluate
        q_value2 = self.q_net2(next_state)
        q_value_evaluate = torch.gather(q_value2.data, dim=1, index=max_action).squeeze()
        target.data.add_(self.gamma * mask * q_value_evaluate)
        out = self.q_net(state)
        y = torch.gather(out, dim=1, index=Variable(action).unsqueeze(1)).squeeze()

        return y, target

    def update(self):
        self.q_net2.load_state_dict(self.q_net.state_dict())

def record_movement(f, env):
    """
    record the movement of rabbit and gun
    :param f: file object
    :param env: environment
    :return: None
    """
    rabbit_loc = env.rabbit_loc
    gun_loc = env.gun_loc

    f.writelines([','.join(map(lambda x: str(x), gun_loc)), '\n'])
    f.writelines([','.join(map(lambda x: str(x), rabbit_loc)), '\n'])


def check_in_corner(env):
    """
    check the rabbit and the gun are in corner
    :param env: environment
    :return: flag
    """
    distance = env.distance

    if distance > 0.1:
        return False

    def in_corner(location):
        return (location[0] <= 0.1*env.screen_size or location[0] >= 0.9*env.screen_size) and \
                (location[1] <= 0.1*env.screen_size or location[1] >= 0.9*env.screen_size)

    return in_corner(env.rabbit_loc) and in_corner(env.gun_loc)

def get_shoot_prob(probnet, location):
    input = Variable(torch.from_numpy(location).unsqueeze(0), volatile=True)
    if use_cuda:
        input = input.cuda()

    pred = probnet(input)
    pred.detach_()
    return pred.data[0, 0]


class HuberLoss(nn.Module):
    def __init__(self):
        super(HuberLoss, self).__init__()

    def forward(self, x):
        square = x**2 * 0.5
        linear = x - 0.5
        mask = (x > 1).float()
        return square * (1.-mask) + linear * mask

class L2Loss(nn.Module):
    def __init__(self):
        super(L2Loss, self).__init__()

    def forward(self, x):
        out = x ** 2 * 0.5
        return out


appended_size = 9

directions = ['NW', 'W', 'SW', 'N', ' ', 'S', 'NE', 'E', 'SE']


def get_obstacle_prediction(state, net):
    """ Pack state as a new state, for general
    """
    input = torch.from_numpy(state[-3:]).unsqueeze(0)
    input = Variable(input, volatile=True)
    output = net(input)
    _, top1_idx = torch.max(output.data, dim=1)
    top1_idx = top1_idx.squeeze()[0]
    return top1_idx


def pack_state(state, obs_direction_idx):
    state[-appended_size + obs_direction_idx] = 1.
    return state


def train_dqn(model, env, render, method, gamma, max_iter,
              batch_size, lr, mem_size, loss,
              init_epsilon, final_epsilon, exploration,
              pretrain, prefix_model_name, prioritized,
              tb_log_dir,
              display_freq=1000, **kwargs):

    saved_model_name = prefix_model_name + '-best-model.pth.tar'
    if use_tensorboard:
        tb_writer = SummaryWriter(tb_log_dir)
    logger.set_logger_dir(tb_log_dir)
    logger.info('logging directory is: {}'.format(tb_log_dir))

    if not os.path.exists('./pretrained'):
        os.mkdir('./pretrained')

    if use_cuda:
        logger.info('Training with GPU')
        model = model.cuda()
    else:
        logger.warning('Training with CPU only')
    # check params
    if method not in METHODS:
        raise NotImplementedError('Unsupported method, availables are: {}'.format(METHODS))

    update_requires = False
    if method in METHODS[1:]:
        if 'update_freq' not in kwargs.keys() or kwargs['update_freq'] <= 0:
            raise RuntimeError('When using method: {}, a positive integer should be provided for updating target network'.
                         format(method))

        update_requires = True
        target_update_freq = kwargs['update_freq']
    # load checkpoint
    best_reward = -10000
    if pretrain is not None:
        if not os.path.exists(pretrain):
            raise RuntimeError('Cannot load pre-trained model: {}'.format(pretrain))
        else:
            logger.info('Train model from checkpoint: {}'.format(pretrain))
            state_dict, info = load_checkpoint(pretrain)
            model.load_state_dict(state_dict)
            for k, v in info.items():
                logger.info('Value of {}: {}'.format(k, v))
            if 'reward' in info.keys():
                best_reward = info['reward']
            else:
                logger.warning('Cannot find reward in checkpoint to update best_reward')
    else:
        logger.info('Train model from scratch')

    eps_schedule = LinearSchedule(exploration, final_epsilon, init_epsilon)

    logger.info('Building memory pool for replay...')
    use_prioritized = False
    if prioritized:
        logger.info('Use prioritized buffer replay')
        if 'alpha' not in kwargs:
            raise KeyError('Should give alpha for prioritized replay')
        if 'beta0' not in kwargs:
            raise KeyError('Should give beta0 for prioritized replay')

        prioritized_alpha = kwargs['alpha']
        prioritized_beta0 = kwargs['beta0']

        replay_buffer = PrioritizedReplayBuffer(mem_size, prioritized_alpha)
        beta_schedule = LinearSchedule(max_iter, initial_p=prioritized_beta0, final_p=1.0)
        use_prioritized = True
    else:
        logger.info('Use uniform sampling buffer replay')
        replay_buffer = ReplayBuffer(mem_size)

    logger.info('Get enough observation for training...')

    step_cnt = 0

    for _ in count(1):
        state = env.reset()
        if render:
            env.render()
        step_cnt += 1
        for _ in count(1):
            action = env.action_space.sample()
            next_state, reward, terminal, _ = env.step(action)
            terminal = int(terminal)

            if render:
                env.render()
            replay_buffer.add(state, next_state, action, reward, terminal)

            state = next_state
            step_cnt += 1
            if terminal or step_cnt >= batch_size * 5:
                break
        if step_cnt >= batch_size * 5:
            break

    logger.info('Begin to train DQN...')
    optimizer = optim.Adam(model.parameters(), lr=lr)

    if not prioritized:
        criterion = nn.SmoothL1Loss() if loss == 'l1' else nn.MSELoss()
    else:
        criterion = HuberLoss() if loss == 'l1' else L2Loss()

    if use_cuda:
        criterion = criterion.cuda()

    if method == 'NIPS':
        helper = Helper(model, gamma)
    elif method == 'Nature':
        helper = NatureHelper(model, gamma)
    elif method == 'Double':
        helper = DoubleHelper(model, gamma)

    best_episode = -1

    cur_episode = 0

    terminal = 1

    moving_loss = 0.
    for step_cnt in xrange(1, max_iter):
        if terminal:
            # a new episode begins!
            cur_episode += 1
            discount = 1.
            bookkeeper = Stat()
            state = env.reset()
            if render:
                env.render()

        optimizer.zero_grad()
        # the agent takes actions and receives (next_state, reward, terminal)
        eps = eps_schedule.value(step_cnt)

        # get action of agent
        if np.random.rand() > eps:
            action, _ = get_action_optimally(model, state)
        # if exploration, then force the action be random
        else:
            action = env.action_space.sample()

        next_state, reward, terminal, info = env.step(action)
        terminal = int(terminal)

        if render:
            env.render()

        if use_tensorboard:
            tb_writer.add_scalar('single_reward', reward, step_cnt)

        # store transition and update state
        replay_buffer.add(state, next_state, action, reward, terminal)
        state = next_state

        # training
        if use_prioritized:
            beta = beta_schedule.value(step_cnt)
            if use_tensorboard:
                tb_writer.add_scalar('prioritized_beta', beta, step_cnt)
            experience = get_batch(replay_buffer, batch_size, use_prioritized, beta)
            bat_state, bat_next_state, bat_action, bat_reward, \
                bat_terminal, weights, bat_batch_idx = experience
        else:
            experience = get_batch(replay_buffer, batch_size, use_prioritized, None)
            bat_state, bat_next_state, bat_action, bat_reward, bat_terminal = experience

        y, target = helper.get_y_target((bat_state, bat_next_state, bat_action, bat_reward, bat_terminal))

        if use_prioritized:
            # loss <- huber_loss(td_error)
            td_error = torch.abs(y-target)
            huber_loss = criterion(td_error)
            weights = Variable(weights)
            weighted_loss = weights * huber_loss
            loss = torch.mean(weighted_loss)

            loss.backward()
            # we need to update priority based on td_error
            new_priority = (td_error.data + 1E-6).cpu().numpy()
            replay_buffer.update_priorities(bat_batch_idx, new_priority)
        else:
            # loss <- nn.SmoothL1Loss(y, target)
            loss = criterion(y, target)
            loss.backward()

        # update the weights
        optimizer.step()
        if use_tensorboard:
            tb_writer.add_scalar('loss', loss.data[0], step_cnt)

        if update_requires and (step_cnt % target_update_freq) == 0:
            helper.update()

        bookkeeper.update(loss=loss.data[0], reward=discount*reward)
        discount *= gamma

        moving_loss = moving_loss * 0.9 + loss.data[0] * 0.1
        if terminal:
            episode_reward = bookkeeper.get_value('reward')
            episode_loss = bookkeeper.get_average('loss')

            if cur_episode % display_freq == 0:
                logger.info('[{:06d}/{:06d}], episode: {:04d}, eps: {:.4f}, loss: {:.4f}, reward: {:.4f}'.format(
                    step_cnt, max_iter, cur_episode, eps, episode_loss, episode_reward))
                test_time_step, test_reward = play(model, env, render, gamma, training=True)
                model_name = os.path.join('./pretrained', prefix_model_name + '-episode-%d.pth.tar' % cur_episode)
                torch.save({'state_dict': model.state_dict(),
                            'reward': test_reward,
                            'time_step': test_time_step,
                            'eps': eps}, model_name)
                logger.info('save checkpoint {} into disk'.format(model_name))
                if use_tensorboard:
                    tb_writer.add_scalar('test-reward', test_reward, step_cnt)
                if test_reward > best_reward:
                    best_reward = test_reward
                    best_episode = cur_episode
                    shutil.copy(model_name, saved_model_name)
                    logger.info('overwrite best model to {}, current best reward: {:.4f} at episode {}'.format(
                        saved_model_name, best_reward, best_episode))
                    if use_tensorboard:
                        tb_writer.add_scalar('best_reward', best_reward, step_cnt)


def play_add_random(model, env,  gamma, control_freq):
    if use_cuda:
        model = model.cuda()

    state = env.reset()
    env.render()
    for step_cnt in count(1):
        if step_cnt % control_freq == 0:
            action, _ = get_action_optimally(model, state)
        else:
            action = np.random.randint(env.get_action_dim())
        state, _, terminal, _ = env.step(action)
        env.render()
        if terminal:
            break


def play(model, env, render, gamma, test_cases=10, training=False):
    if use_cuda:
        model = model.cuda()

    time.sleep(5)
    log = logger.info if training else print
    bookkeepers = []
    for i in xrange(test_cases):
        bookkeeper = Stat()
        discounted = 1.
        state = env.reset()
        if render:
            env.render()
        for _ in count(1):
            action, _ = get_action_optimally(model, state)
            state, reward, terminal, info = env.step(action)
            if render:
                env.render()
            bookkeeper.update(reward=discounted*reward)
            discounted *= gamma
            if terminal:
                break
        log('=>testing: test case: {}, time step: {}, reward: {:.4f}'.format(
            i, bookkeeper.get_time_step(), bookkeeper.get_value('reward')))
        bookkeepers.append(bookkeeper)
    ave_time_step = float(sum([x.get_time_step() for x in bookkeepers])) / test_cases
    ave_reward = sum([x.get_value('reward') for x in bookkeepers]) / test_cases
    log('=>testing, average time step: {:.2f}, reward: {:.4f}'.format(ave_time_step, ave_reward))
    return ave_time_step, ave_reward


def play_cart(model, env, gamma):
    if use_cuda:
        model = model.cuda()
    bookkeeper = Stat()
    for i in xrange(1):
        discounted = 1.
        state = env.reset()
        env.render()
        for _ in count(1):
            action, _ = get_action_optimally(model, state)
            state, reward, terminal, info = env.step(action)
            bookkeeper.update(reward=discounted*reward)
            discounted *= gamma
            env.render()
            if terminal:
                break
    print('reward: {:.2f}, time step: {}'.format(bookkeeper.get_value('reward'), bookkeeper.get_time_step()))
