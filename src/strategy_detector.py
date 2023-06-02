import numpy as np
from player import Player
from constants import ACTIONS, STRATS


class StrategyDetector:
	def __init__(self, games=20):
		self.games = games
		# 0 is cooperate, 1 is defect
		self.strategy = np.concatenate((np.zeros(self.games//2), np.ones(self.games//2)))
		self.change = games//2
		self.player_action_history = None

	def detect(self, player):
		"""Runs a few games to detect the strategy of a player"""
		# Get player's memory capacity
		memory_capacity = player.memory_capacity
		# Initialize opponent history
		self.player_action_history = []
		# Initialize detector action history with n memory_capacity moves
		action_history = np.hstack((np.zeros(memory_capacity), self.strategy)) 
		# Play n games with player 
		for game_i in range(self.games):
			upper = memory_capacity + game_i
			lower = upper - memory_capacity
			history = action_history[lower:upper]
			player_action = player.act(history)
			self.player_action_history.append(player_action)
		# Analyze player strategy
		detected_strategy = self.analyze_player_history()
		return detected_strategy

	def analyze_player_history(self):
		action_history = self.player_action_history
		# Hawk
		if ACTIONS['C'] not in action_history:
			return 2  # Hawk
		# Dove
		elif ACTIONS['D'] not in action_history:
			return 1  # Dove
		# tit for tat
		if action_history[self.change] == ACTIONS['C'] and ACTIONS['D'] not in \
				action_history[:self.change] and ACTIONS['C'] not in action_history[self.change + 1:]:
			return 0  # TitForTat
		# else random
		return 3  # Random/Undetermined

"""
Manual strategies for testing the StrategyDetector
"""

class PlayerStrategy:
	"""
	Abstract class for a player strategy
	"""
	def __init__(self, name):
		self.identifier = name
		self.action_history = []
		self.memory_capacity = 1

	def act(self, opponent):
		pass

	def reset(self):
		self.action_history = []


class TitForTat(PlayerStrategy):
	def __init__(self):
		super().__init__("TitForTat")

	def act(self, history):
		if len(history) == 0:
			self.action_history.append(ACTIONS["C"])
			return ACTIONS["C"]  # cooperate
		else:
			self.action_history.append(history[-1])
			return history[-1]


class Dove(PlayerStrategy):
	def __init__(self):
		super().__init__("Dove")

	def act(self, history):
		self.action_history.append(ACTIONS["C"])
		return ACTIONS["C"]


class Hawk(PlayerStrategy):
	def __init__(self):
		super().__init__("Hawk")

	def act(self, history):
		self.action_history.append(ACTIONS["D"])
		return ACTIONS["D"]


class Random(PlayerStrategy):
	def __init__(self):
		super().__init__("Random")

	def act(self, history):
		self.action_history.append(np.random.choice([0, 1]))
		return self.action_history[-1]


def detect_strategy(player, games=20, verbose=False):
	"""
	Runs a few games to detect the strategy of a player.
	Args:
		player: a Player object
		games: (optional) number of games that are played to detect the strategy
		verbose: whether to print the result
	"""

	detector = StrategyDetector(games=games)

	verdict = detector.detect(player=player)
	if verbose:
		print(f"Player {player.identifier} is seen as: {STRATS[verdict]}")
	return verdict


if __name__ == "__main__":
	for player in [TitForTat(), Dove(), Hawk(), Random()]:
		detect_strategy(player=player, verbose=True)
