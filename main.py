import sys
import os
import random
import numpy as np

script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
module_path = os.path.join(script_directory, 'src')
sys.path.append(module_path)

from environment import Environment as env


if __name__=='__main__':
    n_games = 10
<<<<<<< Updated upstream
    n_matchups = 5
    n_generations = 100
=======
    n_matchups = 10
    n_generations = 1
>>>>>>> Stashed changes

    env = env(n_players=10, n_games=n_games, n_matchups=n_matchups)
    env.run(n_generations=n_generations, verbose=True)





