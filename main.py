# Generic imports
import sys
import os

# Sets correct path for imports
script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
module_path = os.path.join(script_directory, 'src')
sys.path.append(module_path)

# Custom imports
from environment import Environment 
from constants import MAX_MEMORY_CAPACITY
from strategy_detector import StrategyDetector
from player import Player

if __name__ == '__main__':

    # Set simulation parameters
    n_games = 30
    n_matchups = 70
    n_generations = 100
    n_players = 100
    elite = 0.5
    crossover = True
    crossover_p= 0.5
    memory_capacity = 3
    mutation_rate = 0.6

    # Check if parameters are valid
    assert n_players > 1 and n_players % 2 == 0, 'n_players must be even and larger than 1'
    assert memory_capacity <= MAX_MEMORY_CAPACITY and memory_capacity > 0, f'memory_capacity must be between 1 and {MAX_MEMORY_CAPACITY}'
    
    # Create and run simulation
    env = Environment(n_players=n_players, n_games=n_games, n_matchups=n_matchups, mutation_rate=mutation_rate,
                      n_generations=n_generations, elite=elite, crossover=crossover, crossover_p=crossover_p,
                      memory_capacity=memory_capacity)

    env.run(verbose=False)
