# Generic imports
import numpy as np
import matplotlib.pyplot as plt
import os

# Custom imports
from player import Player
from constants import STRATEGY_IDS


class Evaluator:
    """ Creates the functionality to book keep players' rewards, memory capacities, and strategies over generations."""
    def __init__(self, n_players: int, n_generations: int, n_matchups: int, n_games: int, memory_capacity :int, elite: float, 
                 mutation_rate: float, crossover_rate: float, n_simulations: int, payoff_matrix: np.ndarray):

        self.nth_simulation = -1
        shape = (n_simulations, n_players, n_generations)
        self.reward_data = np.zeros(shape)
        self.strategy_data = np.zeros(shape)
        self.environment_data = {'n_players': n_players, 
                                 'n_generations': n_generations, 
                                 'n_matchups': n_matchups, 
                                 'n_games': n_games,
                                 'memory_capacity': memory_capacity,
                                 'payoff_matrix': payoff_matrix}
        self.file_id = f'mc_{memory_capacity}_e_{elite}_mr_{mutation_rate}_cr_{crossover_rate}'
        if not os.path.isdir("./src/figures"):
            os.mkdir("./src/figures")

    def update(self, player: Player, nth_generation: int) -> None:
        """Notifies evaluator of changes in environment state.
        
        Parameters:
        -----------
        player: (Player)
            Player whose data is to be stored.
        
        nth_generation: (int)
            Current generation.
        """ 
        self.reward_data[self.nth_simulation, player.identifier, nth_generation] = player.reward
        self.strategy_data[self.nth_simulation, player.identifier, nth_generation] = player.strategy

    def plot_fitness(self):
        """Plots fitness dynamics of current population over generations."""
        n_generations = self.environment_data['n_generations']
        n_games = self.environment_data['n_games']
        n_matchups = self.environment_data['n_matchups']
        payoff_matrix = self.environment_data['payoff_matrix']
        nth_simulation = self.nth_simulation
        reward_data = self.reward_data[nth_simulation,:,:]
        max_reward = np.max(reward_data, axis=0)
        min_reward = np.min(reward_data, axis=0)
        mean_reward = np.mean(reward_data, axis=0)
        generations = np.arange(n_generations)
        plt.figure(figsize=(10,4))
        plt.subplots_adjust(left=0.1, right=0.8, bottom=0.1, top=0.9, hspace=0.3)
        plt.plot(generations, mean_reward, label='pop. average', color='orange', alpha=1)
        plt.fill_between(generations, min_reward, max_reward, alpha=0.3, color='silver', label='ind. min-max')
        # Maximum reward that can be achieved by an individual
        plt.hlines(y=np.max(payoff_matrix) * n_games * n_matchups, xmin=0, xmax=n_generations-1, 
                   label='ind. optimum', linestyle = '--', color='dodgerblue')
        # Plot all friends fitness threshold
        coop_reward = payoff_matrix[0][0][0]
        plt.hlines(y=coop_reward * n_games * n_matchups, xmin=0, xmax=n_generations-1, 
                   label='pop. optimum', linestyle = '--', color='springgreen')
        # Plot hostile environment threshold
        doubledefect_reward = payoff_matrix[1][1][1]
        plt.hlines(y=doubledefect_reward * n_games * n_matchups, xmin=0, xmax=n_generations-1, 
                   label='pop. minimum', linestyle = '--', color='red')

        plt.title('Fitness of Players over Generations')
        plt.xlabel('nth_generation')
        plt.ylabel('Fitness')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        filepath = f'./src/figures/fitness/fit_sim_{self.nth_simulation}_{self.file_id}.png'
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()

    def plot_average_fitness(self):
        n_generations = self.environment_data['n_generations']
        n_games = self.environment_data['n_games']
        n_matchups = self.environment_data['n_matchups']
        payoff_matrix = self.environment_data['payoff_matrix']
        reward_data = self.reward_data
        max_reward = np.max(reward_data, axis=(1,0), keepdims=True).reshape(-1)
        min_reward = np.min(reward_data, axis=(1,0), keepdims=True).reshape(-1)
        mean_reward = np.mean(reward_data, axis=(0,1), keepdims=True).reshape(-1)
        # Plot min, max, avg of fitness
        generations = np.arange(n_generations)
        plt.figure(figsize=(10,4))
        plt.subplots_adjust(left=0.1, right=0.8, bottom=0.1, top=0.9, hspace=0.3)
        # Maximum reward that can be achieved by an individual
        plt.hlines(y=np.max(payoff_matrix) * n_games * n_matchups, xmin=0, xmax=n_generations-1, 
                   label='ind. optimum', linestyle = '--', color='dodgerblue')
        # Plot all friends fitness threshold
        coop_reward = payoff_matrix[0][0][0]
        plt.hlines(y=coop_reward * n_games * n_matchups, xmin=0, xmax=n_generations-1, 
                   label='pop. optimum', linestyle = '--', color='springgreen')
        plt.plot(generations, mean_reward, label='pop. average', color='orange', alpha=1)
        plt.fill_between(generations, min_reward, max_reward, alpha=0.3, color='silver', label='ind. min-max')
        plt.title('Average Fitness of Players over Generations')
        plt.xlabel('nth_generation')
        plt.ylabel('Fitness')
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        filepath = f'./src/figures/fitness/fit_avg_{self.file_id}.png'
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()

    def plot_strategies(self):
        """ Plots distribution dynamics for a single simulation."""
        plt.figure(figsize=(10,4))
        plt.subplots_adjust(left=0.1, right=0.8, bottom=0.1, top=0.9, hspace=0.3)
        n_generations = self.environment_data['n_generations']
        memory_capacity = self.environment_data['memory_capacity']
        strategy_distributions = np.zeros(shape=(len(STRATEGY_IDS[memory_capacity]), n_generations))
        played_strategies = set()
        for i in range(n_generations):
            strategies = self.strategy_data[self.nth_simulation,:,i]
            for j in range(len(STRATEGY_IDS[memory_capacity])): # reverse order so it lines up with the legend
                indices = np.where(strategies == j)[0]
                if len(indices) > 0:
                    played_strategies.add(j)
                strategy_distributions[j, i] = len(indices)
        strategy_distributions = strategy_distributions[np.any(strategy_distributions !=0, axis=1)]
        plt.stackplot(np.arange(n_generations), strategy_distributions, 
                      labels=list(map(STRATEGY_IDS[memory_capacity].get, list(played_strategies))))
        plt.title(f'Distribution of Strategies over Generations')
        plt.xlabel('nth_generation')
        plt.ylabel(f'Share of Strategies in Population')
        plt.margins(x=0)
        plt.margins(y=0)
        plt.yticks([])
        plt.xticks(np.arange(n_generations+1, step=25))
        # Reverse legend for it to line up with the stackplot
        plt.legend(reversed(plt.legend().legendHandles), reversed(list(map(STRATEGY_IDS[memory_capacity].get, list(played_strategies)))), loc='center left', bbox_to_anchor=(1, 0.5))
        filepath = f'./src/figures/strategies/stgs_sim_{self.nth_simulation}_{self.file_id}.png'
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
    
    def plot_average_strategies(self) -> None:
        """ Plots the average of strategies across multiple runs of the same simulation."""
        plt.figure(figsize=(10,4))
        plt.subplots_adjust(left=0.1, right=0.8, bottom=0.1, top=0.9, hspace=0.3)
        n_players = self.environment_data['n_players']
        memory_capacity = self.environment_data['memory_capacity']
        n_simulations = self.strategy_data.shape[0]
        n_generations = self.strategy_data.shape[2]
        strategy_data = self.strategy_data
        strategy_distributions = np.zeros(shape=(len(STRATEGY_IDS[memory_capacity]), n_generations, n_simulations))
        for i in range(n_simulations):
            for j in range(n_generations):
                for k in range(len(STRATEGY_IDS[memory_capacity])):
                    indices = np.where(strategy_data[i,:,j] == k)[0]
                    strategy_distributions[k, j, i] = len(indices)
        strategy_means = np.mean(strategy_distributions, axis=2)/n_players*100
        # strategy_stds = np.std(strategy_distributions, axis=2)/n_players*100
        # lower = np.maximum(strategy_means - strategy_stds, 0)
        # upper = np.minimum(strategy_means + strategy_stds, 100)
        for i, means in enumerate(strategy_means):
            if memory_capacity == 3:
                if np.max(means) <= 15: # Ignore strategies with peak below 15%
                    continue
            # plt.fill_between(np.arange(n_generations), lower[i,:], upper[i,:], alpha=0.2)
            plt.plot(np.arange(n_generations), means, alpha=0.8, label=STRATEGY_IDS[memory_capacity][i])
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel('nth_generation')
        plt.ylabel('Share of Strategy (%)')
        plt.yticks(np.arange(100+1, step=20))
        plt.xticks(np.arange(n_generations+1, step=25))
        plt.title('Average Distribution of Strategies over Generations')
        filepath = f'./src/figures/strategies/avg_stgs_{self.file_id}.png'
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()


