# Generic imports
import sys
import os
import numpy as np

# Sets correct path for imports
script_directory = os.path.dirname(os.path.abspath(sys.argv[0]))
module_path = os.path.join(script_directory, 'src')
sys.path.append(module_path)

# Custom imports
from environment import Environment 
from constants import MAX_MEMORY_CAPACITY

if __name__ == '__main__':
    # Set simulation parameters
    n_games = 50
    n_matchups = 100
    n_generations = 100
    n_players = 50
    elite = 0.2
    memory_capacity = 3
    mutation_rate = 0.1

    # Check if parameters are valid
    assert n_players >= n_matchups and n_players > 1 and n_players % 2 == 0, 'n_players must be even and larger than 1'
    assert memory_capacity <= MAX_MEMORY_CAPACITY and memory_capacity > 0, f'memory_capacity must be between 1 and {MAX_MEMORY_CAPACITY}'
    
    # Create and run simulation
    env = Environment(n_players=n_players, n_games=n_games, n_matchups=n_matchups, mutation_rate=mutation_rate,
                      n_generations=n_generations, elite=elite, memory_capacity=memory_capacity)

    env.run(verbose=False)



    





