# Generic imports
import sys
import os
from itertools import product
import numpy as np

# Sets correct path for imports
script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
module_path = os.path.join(script_directory, 'src')
sys.path.append(module_path)

# Custom imports
from environment import Environment 
from constants import MAX_MEMORY_CAPACITY
from strategy_detector import StrategyDetector
from player import Player


def run(n_games=50, n_matchups=70, n_generations=100, n_players=100, elite=0.5, memory_capacity=3, mutation_rate=0.5,
        verbose=False, figure_path="./src/figures"):
    # Create and run simulation
    env = Environment(n_players=n_players, n_games=n_games, n_matchups=n_matchups, mutation_rate=mutation_rate,
                      n_generations=n_generations, elite=elite, memory_capacity=memory_capacity, figure_path=figure_path)
    env.run(verbose=verbose)


def permute_rates(step_size_mutation=0.1, step_size_elite=0.1):
    mutations = np.arange(0, 1, step_size_mutation)
    elite_rates = np.arange(step_size_elite, 0.8, step_size_elite)
    for mut_r, elite_r in product(mutations, elite_rates):
        # smaller parameters to decrease run time
        run(n_players=50, n_games=30, n_matchups=30, elite=round(elite_r, 2), mutation_rate=round(mut_r, 2), n_generations=50,
            figure_path="./src/figures/permute_rates")


if __name__ == '__main__':

    # Set simulation parameters
    n_games = 50
    n_matchups = 70
    n_generations = 100
    n_players = 100
    elite = 0.4
    memory_capacity = 2
    mutation_rate = 0.4

    # Check if parameters are valid
    assert n_players > 1 and n_players % 2 == 0, 'n_players must be even and larger than 1'
    assert memory_capacity <= MAX_MEMORY_CAPACITY and memory_capacity > 0, f'memory_capacity must be between 1 and {MAX_MEMORY_CAPACITY}'
    
    # run(n_games=n_games, n_matchups=n_matchups, n_generations=n_generations, n_players=n_players, elite=elite,
    #       memory_capacity=memory_capacity, mutation_rate=mutation_rate, verbose=False)
    permute_rates(step_size_mutation=0.2, step_size_elite=0.2)
