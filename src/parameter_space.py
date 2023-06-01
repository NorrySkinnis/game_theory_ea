import numpy as np
from collections import Counter
from player import Player
from strategy_detector import detect_strategy
from constants import STRATS
import matplotlib.pyplot as plt


def reset_player(player, bias_reset=False):
	player.brain.W1_ = np.zeros_like(player.brain.W1_)
	player.brain.W2_ = np.zeros_like(player.brain.W2_)

	if bias_reset:
		player.brain.Wb1 = np.zeros_like(player.brain.Wb1)
		player.brain.Wb2 = np.zeros_like(player.brain.Wb2)


def print_weights(player, weights=True, bias=True):
	if weights:
		print("W1")
		print(player.brain.W1_)
		print("W2")
		print(player.brain.W2_.T)

	if bias:
		print("Wb1")
		print(player.brain.Wb1)
		print("Wb2")
		print(player.brain.Wb2)


verdicts = []
save = True
verbose = True
players = 1000
for i in range(players):
	if verbose:
		print("-" * 20)
		print(f"Player {i}")
	player = Player(identifier=0, n_matchups=10, n_games=10, memory_capacity=1, use_cuda=False)
	# reset_player(player, bias_reset=True)
	# augment weights here ###

	# player.brain.W1_ = np.array([[1, 1, 1, 1]])
	# player.brain.W2_ = np.array([[1], [1], [1], [1]])
	# player.brain.Wb1 = np.array([[2.0, 2.0, 2.0, 2.0]])
	# player.brain.Wb2 = np.array([[0]])

	####
	print_weights(player, weights=True, bias=True)
	verdict = detect_strategy(player=player, verbose=False)
	verdicts.append(STRATS[verdict])
	if verbose:
		print("-" * 20)

if verbose:
	print(verdicts)
c = Counter(verdicts)

percentages = {}
for key, value in c.items():
	percentage = value/players
	percentages[key] = percentage

if verbose:
	print(percentages)

plt.hist(verdicts)
plt.title(f"Histogram of distribution for {players} players")
plt.xlabel("Strategy")
plt.ylabel("Frequency")

if save:
	plt.savefig('figures/strat_distribution.png')
else:
	plt.show()
