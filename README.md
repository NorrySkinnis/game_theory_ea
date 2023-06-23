Please create env from env.yml to ensure you have the right dependencies.

The main.py file contains everything necessary to run a simulation. Here, you can alter any parameter setting.

n_players: number of individuals in each generation
n_matchups: number of players that each player competes with each generation. Should always be lower than n_players\
n_games: number of games played in each matchup

memory_capacity: size of the input layer of the neural network. Can be run at any size but probably won't produce interesting results beyond size 3\
elite: elitism factor, i.e. the percentage of players that gets to *live* each round\
mutation rate: probability of mutation, per weight per player\
crossover rate: probability of crossover per neuron per individual

n_simulations: how often to repeat simulations with the same settings

Results will be output in data, fitness and strategies folders. Name of file will be combination of the parameter settings.

Changing any other files will void the warranty :)
