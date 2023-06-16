# Generic imports
import sys
import os
from collections import Counter
import itertools
import numpy as np

# Sets correct path for imports
script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
module_path = os.path.join(script_directory, 'src')
sys.path.append(module_path)

# Custom imports
from environment import Environment
from evaluation import Evaluator 


if __name__ == '__main__':

    # Set simulation parameters
    n_players = [50] # must be even
    n_matchups = [30]
    n_games = [20]
    n_generations = [100]

    # Set simulation hyperparameters
    memory_capacity = [3, 2, 3]
    elite = [0.5, 0.5, 0.95]
    mutation_rate = [0.6, 0.5, 1]
    crossover = [True]
    crossover_p= [0.2, 0.3, 0.6]

    # Create permutations of settings
    sim_settings = itertools.product(n_players, n_matchups, n_games, n_generations, 
                                     memory_capacity, elite, mutation_rate, crossover, 
                                     crossover_p)
    
    repetitions = 5
    for setting in sim_settings:
        s = []
        for _ in range(repetitions):
            simulation = Environment(*setting)
            simulation.run(verbose=False)
            strategy_data = simulation.evaluator.strategy_data
            s.append(strategy_data)
        Evaluator.plot_average_strategies(s)
        break
        

    # Used printing out the codes of the undetermined strategies

    # unknown = env.detector.undetermined_strategies
    # strategy_count = Counter(frozenset(s) for s in unknown)
    # print(strategy_count)

