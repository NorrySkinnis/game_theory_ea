import sys
import os

script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
module_path = os.path.join(script_directory, 'src')
sys.path.append(module_path)

from environment import Environment as env


if __name__=='__main__':
    n_games = 1000
    n_matchups = 100
    n_generations = 10

    env = env(n_players=10)
    players = env.run(n_games=n_games, 
                      n_matchups=n_matchups, 
                      n_generations=n_generations)