有各式各样的环境形式， 但基本操作是相同的。所以， 要有一个基本的功能代码， 其它的就看各种实现形式了。

basic.py                    game_engine.py                 game_general.py    
game_keepdistance.py        game_keephitrate.py   
game_maze_competition.py    game_maze_hiding_cheesing.py   game_maze_hiding.py    game_maze_keephitrate_cheesing.py    game_maze_keephitrate_multi_shooter.py   game_maze_keephitrate.py   
game_maze.py  
game_semicircular.py   
game_smater_gun.py   
__init__.py   
maze_env.py                 maze.py                         maze_world.py


/util/common.py :
1. 而真正需要做的是设置游戏模式MODES
2. 还有就是根据游戏引擎（game engine）获取状态和动作的维数， 作为神经网络的输入和输出。
也就是 ninp = env.get_state_dim()、nout = env.get_action_dim()是环境需要提供的。

 MODES = ['keep-distance', 'keep-hit-rate', 'maze-hiding', 'maze-keep-hit-rate', 'general-distance',
           'maze-keep-hit-rate-cheesing', 'maze-hiding-cheesing', 'general-maze-keep-hit-rate',
           'general-another-maze-keep-hit-rate', 'general-maze-keep-distance',
           'general-another-maze-keep-distance', 'general-maze-keep-hit-rate2',
           'general-another-maze-keep-hit-rate2',
           'general-maze-keep-hit-rate3', 'general-another-maze-keep-hit-rate3',
           'semi-circle-keephitrate', 'maze-smarter-gun',
           'maze-competition', 'maze-keep-hit-rate-multi-shooter']

每种模式对应着一种环境， 共19种，而体现模式之间的不同的是参数设置的不同：
1.  mode == 'keep-distance'                        env = GameKeepDistance(**params)                           <---  (game_keepdistance.py)
2.  mode == 'general-distance'                     env = GameGeneral(**params)                                <---  (game_general.py)
3.  mode == 'keep-hit-rate'                        env = GameKeepHitRate(**params)                            <---  (game_keephitrate.py)
4.  mode == 'maze-hiding'                          env = GameMazeHiding(**params)                             <---  (game_maze_hiding.py)
5.  mode == 'maze-keep-hit-rate'                   env = GameMazeKeepHitRate(**params)                        <---  (game_maze_keephitrate.py)
6.  mode == 'maze-hiding-cheesing'                 env = GameMazeHidingCheesing(**params)                     <---  (game_maze_hiding_cheesing.py)
7.  mode == 'maze-keep-hit-rate-cheesing'          env = GameMazeKeepHitRateCheesing(**params)                <---  (game_maze_keephitrate_cheesing.py)
8.  mode == 'general-maze-keep-hit-rate'           env = GameMazeGeneralKeepHitRate(**params)                 <---  (game_general.py)
9.  mode == 'general-another-maze-keep-hit-rate'   env = GameAnotherMazeGeneralKeepHitRate(**params)          <---  (game_general.py)
10. mode == 'general-maze-keep-distance'           env = GameMazeGeneralKeepDistance(**params)                <---  (game_general.py)
11. mode == 'general-another-maze-keep-distance'   env = GameAnotherMazeGeneralKeepDistance(**params)         <---  (game_general.py)
12. mode == 'general-maze-keep-hit-rate2'          env = GameMazeGeneralKeepHitRate2(**params)                <---  (game_general.py)
13. mode == 'general-another-maze-keep-hit-rate2'  env = GameAnotherMazeGeneralKeepHitRate2(**params)         <---  (game_general.py)
14. mode == 'general-maze-keep-hit-rate3'          env = GameMazeGeneralKeepHitRate3(**params)                <---  (game_general.py)
15. mode == 'general-another-maze-keep-hit-rate3'  env = GameAnotherMazeGeneralKeepHitRate3(**params)         <---  (game_general.py)
16. mode == 'semi-circle-keephitrate'              env = GameSemiCircular(**params)                           <---  (game_semicircular.py)     
17. mode == 'maze-smarter-gun'                     env = GameSmartGun(**params)                               <---  (game_smater_gun.py)
18. mode == 'maze-keep-hit-rate-multi-shooter'     env = GameMazeKeepHitRateMultiShooter(**params)            <---  (game_maze_keephitrate_multi_shooter.py)
19. mode == 'maze-competition'                     env = GameMazeKeepHitRate(**params)    [ env = GameMazeCompetition(**params) (兔子模型) ]  <---  (game_maze_competition.py)
