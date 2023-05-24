import sys
import os
import random
import numpy as np

script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
module_path = os.path.join(script_directory, 'src')
sys.path.append(module_path)

from environment import Environment as env


if __name__=='__main__':
    n_games = 100
    n_matchups = 100
    n_generations = 100

    env = env(n_players=100, n_games=n_games, n_matchups=n_matchups, n_generations=n_generations)
    env.run(verbose=False)
    env.evaluater.plot_fitness(max=True, min=True)





