import numpy as np

from player import Player
from strategy_detector import detect_strategy
from constants import STRATS


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
for i in range(1):
	print("-" * 20)
	print(f"Player {i}")
	player = Player(identifier=0, n_matchups=10, n_games=10, memory_capacity=1, use_cuda=False)
	reset_player(player, bias_reset=True)
	# augment weights here ###

	player.brain.W1_ = np.array([[0, 1, 0, 0]])
	player.brain.W2_ = np.array([[0], [-1], [0], [0]])
	# player.brain.Wb1 = np.array([[2.0, 2.0, 2.0, 2.0]])
	player.brain.Wb2 = np.array([[0]])

	####
	print_weights(player, weights=True, bias=True)
	verdict = detect_strategy(player=player, verbose=False)
	verdicts.append(STRATS[verdict])
	print("-" * 20)

print(verdicts)
