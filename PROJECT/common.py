import os
import time
from model import *
from game import *

from .nn_misc import load_checkpoint

import torch

MODES = ['keep-distance', 'keep-hit-rate', 'maze-hiding', 'maze-keep-hit-rate', 'general-distance',
         'maze-keep-hit-rate-cheesing', 'maze-hiding-cheesing', 'general-maze-keep-hit-rate',
         'general-another-maze-keep-hit-rate', 'general-maze-keep-distance',
         'general-another-maze-keep-distance', 'general-maze-keep-hit-rate2',
         'general-another-maze-keep-hit-rate2',
         'general-maze-keep-hit-rate3', 'general-another-maze-keep-hit-rate3',
         'semi-circle-keephitrate', 'maze-smarter-gun',
         'maze-competition', 'maze-keep-hit-rate-multi-shooter']

GYM_MODES = ['cartpole']

# get model files
path = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, 'model'))
py_files = filter(lambda x: os.path.splitext(x)[-1] == '.py', os.listdir(path))
model_files = filter(lambda x: os.path.splitext(x)[0] != '__init__', py_files)
MODELS = [os.path.splitext(x)[0] for x in model_files]

# methods for dqn
METHODS = ['NIPS', 'Nature', 'Double']


def model_factory(args, env):
    """
    Generate model given argument and the environment.
    """
    model_type = args.model_type.lower()
    mode = args.mode.lower()

    if mode in MODES:
        ninp = env.get_state_dim()
        nout = env.get_action_dim()
    elif mode in GYM_MODES:
        nout = env.action_space.n
        ninp = env.observation_space.shape[0]
    else:
        raise RuntimeError('Unsupported Game Mode')

    if model_type == 'cnn':
        in_channels = 1
        convs = [(8, 4, 1, 0), (16, 4, 2, 1), (32, 4, 2, 1), (64, 4, 2, 1)]
        nproj = 9*9*64
        model = CNN(in_channels, convs, nproj, nout)

    elif model_type == 'ann':
        nhids = args.hidden_size
        model = ANN(ninp, nout, nhids)

    elif model_type == 'rnn_policy':
        rnn_type = args.rnn_type.upper()
        nhids = args.hidden_size
        model = RNNPolicy(rnn_type, ninp, nout, nhids[0])

    elif model_type == 'ann_policy':
        nhids = args.hidden_size
        model = ANNPolicy(ninp, nout, nhids[0])

    return model


def env_factory(args):
    mode = args.mode.lower()
    if mode not in MODES and mode not in GYM_MODES:
        raise RuntimeError('Invalid mode, available modes are: {}, {}'.format(MODES, GYM_MODES))

    if mode in GYM_MODES:
        import gym
        if mode == 'cartpole':
            env = gym.make('CartPole-v0')
            return env

    params = {'training': args.train,
              'screen_size': args.screen_size,
              'rabbit_size': args.rabbit_size,
              'rabbit_speed': args.rabbit_speed,
              'gun_size': args.gun_size,
              'gun_speed': args.gun_speed,
              'max_time_step': args.max_time_step,
              'history_size': args.history_size,
              'debug': False}

    if mode == 'keep-distance':
        params['desired_goal'] = args.desired_goal
        params['lower_goal'] = args.lower_goal
        params['higher_goal'] = args.higher_goal
        env = GameKeepDistance(**params)

    elif mode == 'general-distance':
        params['desired_goal'] = args.desired_goal
        params['lower_goal'] = args.lower_goal
        params['higher_goal'] = args.higher_goal
        env = GameGeneral(**params)

    elif mode == 'keep-hit-rate':
        params['desired_goal'] = args.desired_goal
        env = GameKeepHitRate(**params)

    elif mode == 'maze-hiding':
        params['obstacle_size'] = 0.2
        env = GameMazeHiding(**params)

    elif mode == 'maze-keep-hit-rate':
        params['obstacle_size'] = 0.2
        params['desired_goal'] = args.desired_goal
        params['transparency'] = args.transparency
        env = GameMazeKeepHitRate(**params)

    elif mode == 'maze-hiding-cheesing':
        params['obstacle_size'] = 0.2
        env = GameMazeHidingCheesing(**params)

    elif mode == 'maze-keep-hit-rate-cheesing':
        params['obstacle_size'] = 0.2
        params['desired_goal'] = args.desired_goal
        env = GameMazeKeepHitRateCheesing(**params)

    elif mode == 'general-maze-keep-hit-rate':
        params['obstacle_size'] = 0.2
        params['desired_goal'] = args.desired_goal
        env = GameMazeGeneralKeepHitRate(**params)

    elif mode == 'general-another-maze-keep-hit-rate':
        params['obstacle_size'] = 0.2
        params['desired_goal'] = args.desired_goal
        env = GameAnotherMazeGeneralKeepHitRate(**params)

    elif mode == 'general-maze-keep-distance':
        params['obstacle_size'] = 0.2
        params['desired_goal'] = 0.3
        env = GameMazeGeneralKeepDistance(**params)

    elif mode == 'general-another-maze-keep-distance':
        params['obstacle_size'] = 0.2
        params['desired_goal'] = 0.3
        env = GameAnotherMazeGeneralKeepDistance(**params)

    elif mode == 'general-maze-keep-hit-rate2':
        params['obstacle_size'] = 0.2
        params['desired_goal'] = args.desired_goal
        env = GameMazeGeneralKeepHitRate2(**params)

    elif mode == 'general-another-maze-keep-hit-rate2':
        params['obstacle_size'] = 0.2
        params['desired_goal'] = args.desired_goal
        env = GameAnotherMazeGeneralKeepHitRate2(**params)

    elif mode == 'general-maze-keep-hit-rate3':
        params['obstacle_size'] = 0.2
        params['desired_goal'] = args.desired_goal
        env = GameMazeGeneralKeepHitRate3(**params)

    elif mode == 'general-another-maze-keep-hit-rate3':
        params['obstacle_size'] = 0.2
        params['desired_goal'] = args.desired_goal
        env = GameAnotherMazeGeneralKeepHitRate3(**params)
        
    elif mode == 'semi-circle-keephitrate':
        params['desired_goal'] = args.desired_goal
        params['transparency'] = args.transparency
        env = GameSemiCircular(**params)

    elif mode == 'maze-smarter-gun':
        params['obstacle_size'] = 0.2
        params['desired_goal'] = args.desired_goal
        params['transparency'] = args.transparency
        env = GameSmartGun(**params)

    elif mode == 'maze-keep-hit-rate-multi-shooter':
        params['obstacle_size'] = 0.2
        params['desired_goal'] = args.desired_goal
        params['transparency'] = args.transparency
        env = GameMazeKeepHitRateMultiShooter(**params)

    elif mode == 'maze-competition':
        params['obstacle_size'] = 0.2
        params['desired_goal'] = args.desired_goal
        params['transparency'] = args.transparency
        env = GameMazeKeepHitRate(**params)
        # load the rabbit model
        model = model_factory(args, env)
        pretrained = args.rabbit_init_model
        if not os.path.exists(pretrained):
            raise RuntimeError('cannot find rabbit init model: {}'.format(
                pretrained))
        state_dict, _ = load_checkpoint(pretrained)
        model.load_state_dict(state_dict)

        if torch.cuda.is_available():
            model = model.cuda()

        params['rabbit_init_model'] = model
        env = GameMazeCompetition(**params)


    print('Game Environment Configuration:')
    for k, v in params.items():
        print('{}: {}'.format(k, v))

    return env


def get_current_time(fmt='%H-%M-%S@%Y-%m-%d'):
    return time.strftime(fmt, time.localtime())
