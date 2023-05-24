import numpy as np


class StrategyDetector:
	def __init__(self, possible_strats, games=10):
		self.games = games
		self.detection_strategy = ["C", "C", "C", "C", "C", "D", "D", "D", "D", "D"]
		self.change = 5
		self.possible_strats = possible_strats
		self.action_history = []

	def detect(self, player):
		"""Runs a few games to detect the strategy of a player"""
		player.clear_history()
		opponent_history = []
		for g in range(self.games):
			my_action = self.detection_strategy[g]
			opponent_history.append(player.choose_action(self))
			self.action_history.append(my_action)
		verdict = self.analyze_history(opponent_history)
		self.action_history = []
		return verdict

	def analyze_history(self, opponent_history):
		if 'C' not in opponent_history:
			return "Hawk"
		elif 'D' not in opponent_history:
			return "Dove"
		# tit for tat
		if opponent_history[self.change] == 'C' and 'D' not in opponent_history[:self.change] and \
			'C' not in opponent_history[self.change+1:]:
			return 'Tit for Tat'
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

	def choose_action(self, opponent):
		pass

	def reset(self):
		self.action_history = []


class TitForTat(PlayerStrategy):
	def __init__(self):
		super().__init__("TitForTat")

	def choose_action(self, opponent):
		if len(opponent.history) == 0:
			self.action_history.append("C")
			return "C"
		else:
			self.action_history.append(opponent.history[-1])
			return opponent.history[-1]


class Dove(PlayerStrategy):
	def __init__(self):
		super().__init__("Dove")

	def choose_action(self, opponent):
		self.action_history.append("C")
		return "C"


class Hawk(PlayerStrategy):
	def __init__(self):
		super().__init__("Hawk")

	def choose_action(self, opponent):
		self.action_history.append("D")
		return "D"


class Random(PlayerStrategy):
	def __init__(self):
		super().__init__("Random")

	def choose_action(self, opponent):
		self.action_history.append(np.random.choice(["C", "D"]))
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
