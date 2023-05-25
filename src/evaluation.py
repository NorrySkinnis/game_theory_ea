from player import Player
import numpy as np
import matplotlib.pyplot as plt

class Evaluater:

    def __init__(self, players:list[Player], n_generations:int):
        self.n_generations = n_generations
        self.rewards_per_gen = np.zeros(shape=(len(players), n_generations))
        self.memory_capacities_per_gen = np.zeros(shape=(len(players), n_generations))

    def update(self, player:Player, nth_generation:int)->None:
        """Trigger snapshot of players' rewards and memory capacities at nth generation.""" 
        self.rewards_per_gen[player.identifier,nth_generation] = player.reward
        self.memory_capacities_per_gen[player.identifier,nth_generation] = player.memory_capacity

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