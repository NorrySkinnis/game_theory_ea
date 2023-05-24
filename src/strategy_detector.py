import numpy as np
from player import Player


actions_dict = {
				"C": 0,
				"D": 1
				}


class StrategyDetector:
	def __init__(self, possible_strats, games=10):
		self.games = games
		# 0 is cooperate, 1 is defect
		self.detection_strategy = np.concatenate((np.zeros(self.games//2), np.ones(self.games//2)))
		self.change = games//2
		self.possible_strats = possible_strats
		self.action_history = []  # gets initialised and modified in detect function


	def detect(self, player):
		"""Runs a few games to detect the strategy of a player"""
		# player.reset()
		# prepare action history based on memory capacity of opponent
		self.action_history = -np.ones(shape=(1, self.games + player.memory_capacity), dtype=int)
		pre_cooperate = np.zeros(shape=(1, player.memory_capacity))
		self.action_history[:, :player.memory_capacity] = pre_cooperate  # prepends actions to history

		opponent_history = []
		for g in range(self.games):
			my_action = self.detection_strategy[g]
			upper = Player.max_memory_capacity + g
			lower = upper - player.memory_capacity
			self.action_history[:, upper] = my_action
			history = self.action_history[:, lower:upper]
			opponent_history.append(player.act(history))

		verdict = self.analyze_history(opponent_history)
		self.action_history = []
		return verdict

	def analyze_history(self, opponent_history):
		# Hawk
		if actions_dict['C'] not in opponent_history:
			return "Hawk"
		# Dove
		elif actions_dict['D'] not in opponent_history:
			return "Dove"
		# tit for tat
		if opponent_history[self.change] == actions_dict['C'] and actions_dict['D'] not in \
				opponent_history[:self.change] and actions_dict['C'] not in opponent_history[self.change+1:]:
			return 'Tit for Tat'
		# else random
		return 'Random/Undetermined'


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
			self.action_history.append(actions_dict["C"])
			return actions_dict["C"]  # cooperate
		else:
			self.action_history.append(history[-1])
			return history[-1]


class Dove(PlayerStrategy):
	def __init__(self):
		super().__init__("Dove")

	def act(self, history):
		self.action_history.append(actions_dict["C"])
		return actions_dict["C"]


class Hawk(PlayerStrategy):
	def __init__(self):
		super().__init__("Hawk")

	def act(self, history):
		self.action_history.append(actions_dict["D"])
		return actions_dict["D"]


class Random(PlayerStrategy):
	def __init__(self):
		super().__init__("Random")

	def act(self, history):
		self.action_history.append(np.random.choice([0, 1]))
		return self.action_history[-1]


def detect_strategy(player, verbose=False):
	"""
	Runs a few games to detect the strategy of a player.
	Args:
		player: a Player object
		verbose: whether to print the result
	"""

	possible_strats = [TitForTat(), Dove(), Hawk(), Random()]
	detector = StrategyDetector([c.identifier for c in possible_strats])

	verdict = detector.detect(player=player)
	if verbose:
		print(f"Player {player.identifier} is seen as: {verdict}")
	return verdict


if __name__ == "__main__":
	for player in [TitForTat(), Dove(), Hawk(), Random()]:
		detect_strategy(player=player, verbose=True)
