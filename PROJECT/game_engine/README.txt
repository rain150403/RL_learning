有各式各样的环境形式， 但基本操作是相同的。所以， 要有一个基本的功能代码， 其它的就看各种实现形式了。

basic.py                    game_engine.py
game_general.py  （多次使用）
game_maze.py（修改之后也能运行的一种游戏模式）

game_keepdistance.py        game_keephitrate.py   
game_maze_competition.py    game_maze_hiding_cheesing.py   game_maze_hiding.py    game_maze_keephitrate_cheesing.py    game_maze_keephitrate_multi_shooter.py   game_maze_keephitrate.py   
game_semicircular.py   
game_smater_gun.py   



maze_env.py                 maze.py                        maze_world.py

__init__.py   (最最基本)

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



__init__.py的作用有如下几点：
  1. 相当于class中的def __init__(self):函数，用来初始化模块。
  2. 把所在目录当作一个package处理
  3. from-import 语句导入子包时需要用到它。 如果没有用到, 他们可以是空文件。
       如引入package.module下的所有模块
       from package.module import * 
       这样的语句会导入哪些文件取决于操作系统的文件系统. 所以我们在__init__.py 中加入 __all__变量. 
       该变量包含执行这样的语句时应该导入的模块的名字. 它搜索由一个模块名字符串列表组成.
