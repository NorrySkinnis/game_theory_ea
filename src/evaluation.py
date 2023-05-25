from player import Player
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from constants import STRATS


class Evaluator:

    def __init__(self, players:list[Player], n_generations:int):
        self.n_generations = n_generations
        self.players = players
        self.rewards_per_gen = np.zeros(shape=(len(players), n_generations))
        self.memory_capacities_per_gen = np.zeros(shape=(len(players), n_generations))
        self.strategy_data = np.empty(shape=(self.n_generations, len(players)))

    def update(self, player:Player, nth_generation:int, verdict:int)->None:
        """Trigger snapshot of players' rewards and memory capacities at nth generation.""" 
        self.rewards_per_gen[player.identifier,nth_generation] = player.reward
        self.memory_capacities_per_gen[player.identifier,nth_generation] = player.memory_capacity
        self.strategy_data[nth_generation, player.identifier] = verdict

    def plot_fitness(self, max=True, min=True):
        gens = np.arange(1, self.n_generations+1)
        if max:
            max_reward = np.max(self.rewards_per_gen, axis=0)
        if min:
            min_reward = np.min(self.rewards_per_gen, axis=0)
        mean_reward = np.mean(self.rewards_per_gen, axis=0)
        plt.plot(gens, mean_reward, label='Avg')
        plt.plot(gens, max_reward, label='Max')
        plt.plot(gens, min_reward, label='Min')
        plt.legend()
        plt.show()

    def plot_strategies(self, name=None, title='', save=True):
        """
        Plot the distribution of strategies over generations.

        Args:
            name: (optional) name of the file to save the figure
            title: (optional) title of the figure
            save: (optional) if True, save the figure to 'figures/' directory
        """
        plt.figure()
        generations = np.arange(self.strategy_data.shape[0])
        y = [[] for _ in STRATS.keys()]
        for gen_data in self.strategy_data:
            c = Counter(gen_data)
            for k in STRATS.keys():
                if k not in c.keys():
                    c[k] = 0
            for k in c.keys():
                y[int(k)].append(c[int(k)])

        plt.stackplot(generations, y)
        plt.title(title)
        plt.xlabel('Generations')
        plt.ylabel('Relative strategy distribution (%)')  # make labels so it lines up with colour
        plt.legend([STRATS[k] for k in STRATS.keys()])
        plt.xticks(np.arange(0, self.n_generations+self.n_generations//10, self.n_generations//10))
        plt.margins(x=0)
        plt.margins(y=0)

        if save:
            if name:
                plt.savefig('src/figures/' + name + '.png', bbox_inches='tight')
            else:
                plt.savefig(f'src/figures/stackplot_g{self.n_generations}_p{len(self.players)}.png',
                            bbox_inches='tight')
        else:
            plt.show()
