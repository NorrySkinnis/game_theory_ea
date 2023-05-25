from player import Player
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


class Evaluator:

    def __init__(self, players:list[Player], n_generations:int):
        self.n_generations = n_generations
        self.rewards_per_gen = np.zeros(shape=(len(players), n_generations))
        self.memory_capacities_per_gen = np.zeros(shape=(len(players), n_generations))
        self.strats = {0: 'TitForTat', 1: "Dove", 2: "Hawk", 3: "Random"}
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

    def plot_strategies(self, data, name, title='', save=True):
        plt.figure()
        generations = np.arange(data.shape[0])
        y = [[] for i in self.strats.keys()]
        for gen_data in data:
            c = Counter(gen_data)
            for k in self.strats.keys():
                if k not in c.keys():
                    c[k] = 0
            for k in c.keys():
                y[k].append(c[k])

        # print(generations.shape)
        # print(y.shape)
        plt.stackplot(generations, y)
        plt.title(title)
        plt.xlabel('Generations')
        plt.ylabel('Relative strategy distribution (%)')  # make labels so it lines up with colour
        plt.legend([strats[k] for k in strats.keys()])

        if save:
            plt.savefig('figures/' + name + '.png')
        else:
            plt.show()