# Generic imports
import numpy as np
import matplotlib.pyplot as plt
import os

# Custom imports
from player import Player
from constants import STRATEGY_IDS


class Evaluator:
    """ Creates the functionality to book keep players' rewards, memory capacities, and strategies over generations."""

    def __init__(self, players:list[Player], n_generations:int, n_games:int, n_matchups:int, memory_capacity:int,
                 mutation_rate:float, payoff_matrix:np.ndarray):
        self.players = players
        self.n_generations = n_generations
        self.n_games = n_games
        self.n_matchups = n_matchups
        self.payoff_matrix = payoff_matrix
        self.mutation_rate = mutation_rate
        self.memory_capacity = memory_capacity
        
        # Initializes data structure for book keeping
        shape = (len(players), n_generations)
        self.rewards_per_gen = np.zeros(shape)
        self.strategy_data = np.zeros(shape)
        self.memory_capacities_per_gen = np.zeros(shape)

        if not os.path.isdir("./src/figures"):
            os.mkdir("./src/figures")

    def update(self, player: Player, nth_generation: int, player_strategy: int) -> None:
        """Triggers snapshot of players' rewards, memory capacities, strategy at nth generation.
        
        Parameters:
        -----------
        player: (Player)
            Player whose data is to be stored.
        
        nth_generation: (int)
            Current generation.
        
        player_strategy: (int)
            Id of player's strategy.
        """ 
        self.rewards_per_gen[player.identifier,nth_generation] = player.reward
        self.strategy_data[player.identifier, nth_generation] = player_strategy
        self.memory_capacities_per_gen[player.identifier,nth_generation] = player.memory_capacity

    def plot_fitness(self):
        """Plots fitness dynamics after simulation is finished."""
        gens = np.arange(self.n_generations)
        max_reward = np.max(self.rewards_per_gen, axis=0)
        min_reward = np.min(self.rewards_per_gen, axis=0)
        mean_reward = np.mean(self.rewards_per_gen, axis=0)
        legend = []
        # Plot min, max, avg of fitness
        plt.figure()
        plt.plot(gens, max_reward, label='Max', c='m')
        legend.append('max')
        plt.plot(gens, mean_reward, label='Avg', c='r')
        legend.append('avg')
        plt.plot(gens, min_reward, label='Min', c='m')
        legend.append('min')
        # Plot max theoretical fitness
        betray_reward = np.max(self.payoff_matrix)
        plt.hlines(y=betray_reward * self.n_games * self.n_matchups, xmin=0, xmax=self.n_generations-1, linestyle = '--', color='gray')
        legend.append('theoretical max')
        # Plot all friends fitness threshold
        coop_reward = self.payoff_matrix[0][0][0]
        plt.hlines(y=coop_reward * self.n_games * self.n_matchups, xmin=0, xmax=self.n_generations-1, linestyle = '-.', color='gray')
        legend.append('all C threshold')
        plt.title('Fitness of Players over Generations')
        plt.xlabel('nth_generation')
        plt.ylabel('Fitness')
        plt.legend(legend)
        plt.savefig(f'src/figures/fitness_gen{self.n_generations}_p{len(self.players)}_m{self.n_matchups}'
            f'_g{self.n_games}_mem{self.memory_capacity}_mut{self.mutation_rate}.png', bbox_inches='tight')

    def plot_strategies(self):
        """ Plot the distribution of strategies over generations."""
        plt.figure()
        n_generations = self.n_generations
        strategy_distributions = np.zeros(shape=(len(STRATEGY_IDS), n_generations))
        for i in range(n_generations):
            strategies = self.strategy_data[:,i]
            for j in range(len(STRATEGY_IDS)):
                indices = np.where(strategies == j)[0]
                strategy_distributions[j, i] = len(indices)
        plt.stackplot(np.arange(n_generations), strategy_distributions, labels=STRATEGY_IDS.values())
        plt.title(f'Distribution of Strategies over Generations')
        plt.xlabel('nth_generation')
        plt.ylabel(f'Share of Strategies in Population')
        plt.margins(x=0)
        plt.margins(y=0)
        plt.legend()
        plt.savefig(f'src/figures/strats_gen{self.n_generations}_p{len(self.players)}_m{self.n_matchups}'
        f'_g{self.n_games}_mem{self.memory_capacity}_mut{self.mutation_rate}.png',
                    bbox_inches='tight')
