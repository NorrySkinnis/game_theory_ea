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
from strategy_detector import StrategyDetector


if __name__ == '__main__':

    # Set payoff matrix and fitness function
    payoff_matrix = np.array([[(3, 3), (0, 5)], [(5, 0), (1, 1)]])
    fitness = lambda x, t: np.power(1, t) * np.sum(x)

    # Set simulation parameters
    n_players = 100 # must be even
    n_generations = 150
    n_matchups = 60
    n_games = 15

    # Set simulation hyperparameters
    memory_capacity = [1, 2]
    elite = [0.1, 0.5, 0.95]
    mutation_rate = [0.1, 0.5, 1]
    crossover_rate = [0, 0.3, 0.6]

    # memory_capacity = [2]
    # elite = [0.5]
    # mutation_rate = [0.1]
    # crossover_rate = [0] 

    # Create permutations of settings
    sim_settings = itertools.product([n_players], [n_generations], [n_matchups], [n_games],
                                     memory_capacity, elite, mutation_rate, crossover_rate)
    
    # Iterate over simulation settings and run simulations n times
    n_simulations = 1
    for setting in sim_settings:

        evaluator = Evaluator(*setting, n_simulations=n_simulations, payoff_matrix=payoff_matrix)
        
        for _ in range(n_simulations):

            simulation = Environment(*setting, payoff_matrix=payoff_matrix, fitness=fitness)
            simulation.attach(evaluator)
            simulation.run(verbose=False)
            # Get codes of undetermined strategies
            unknown = simulation.detector.undetermined_strategies
            strategy_count = Counter(frozenset(s) for s in unknown)
            print(strategy_count)
            evaluator.plot_fitness()
            evaluator.plot_strategies()

        evaluator.plot_average_fitness()
        evaluator.plot_average_strategies()
        # break

