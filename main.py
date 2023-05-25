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
    n_games = 3
    n_matchups = 5
    n_generations = 1
    n_players = 1
    verbose = False
    memory_capacity = 1

    for i, arg in enumerate(sys.argv):
            n_matchups = int(sys.argv[i+1])
        elif arg == '-matchups':
            n_games = int(sys.argv[i+1])
        elif arg == '-games':
            n_players = int(sys.argv[i+1])
        if arg == '-players':
        elif arg == '-generations':
            n_generations = int(sys.argv[i+1])
        elif arg == '-v':
            verbose = True
        elif arg == '-mem_global':
            memory_capacity = int(sys.argv[i+1])
            
    env = env(n_players=n_players, n_games=n_games, n_matchups=n_matchups, n_generations=n_generations, memory_capacity=memory_capacity)
    env.run(verbose=verbose)
    env.evaluater.plot_fitness(max=True, min=True)
    





