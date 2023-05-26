import sys
import os
import random
import numpy as np

script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
module_path = os.path.join(script_directory, 'src')
sys.path.append(module_path)

from environment import Environment as env


if __name__ == '__main__':
    # use command line args as follows:
    # python main.py -players 1 -games 2 -generations 3 -matchups 4 -v

    # default values if not command line args are given
    n_games = 100
    n_matchups = 50
    n_generations = 3
    n_players = 10  # at least 2
    memory_capacity = 1  # at least 1
    verbose = False
    strat_detector = True
    use_cuda = False

    for i, arg in enumerate(sys.argv):
        if arg == '-players':
            n_players = int(sys.argv[i + 1])
            if n_players < 2:
                raise ValueError('Number of players must be at least 2')
        elif arg == '-matchups':
            n_matchups = int(sys.argv[i + 1])
        elif arg == '-games':
            n_games = int(sys.argv[i + 1])
        elif arg == '-generations':
            n_generations = int(sys.argv[i+1])
        elif arg == '-v':
            verbose = True
        elif arg == '-mem_global':
            memory_capacity = int(sys.argv[i+1])
            if memory_capacity < 1:
                raise ValueError('Memory capacity must be at least 1')
            
    env = env(n_players=n_players, n_games=n_games, n_matchups=n_matchups, n_generations=n_generations,
              memory_capacity=memory_capacity, strat_detector=strat_detector, use_cuda=use_cuda)
    env.run(verbose=verbose)
    env.evaluator.plot_fitness(max=True, min=True)
    if strat_detector:
        env.evaluator.plot_strategies(title=f'Strategies over generations, gen={n_generations}, players={n_players}')
    





