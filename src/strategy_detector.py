import numpy as np


class StrategyDetector:
	def __init__(self, possible_strats, games=10):
		self.games = games
		self.detection_strategy = ["C", "C", "C", "C", "C", "D", "D", "D", "D", "D"]
		self.change = 5
		self.possible_strats = possible_strats
		self.history = []

	def detect(self, player):
		"""Runs a few games to detect the strategy of a player"""
		player.clear_history()
		opponent_history = []
		for g in range(self.games):
			my_action = self.detection_strategy[g]
			opponent_history.append(player.choose_action(self))
			self.history.append(my_action)
		verdict = self.analyze_history(opponent_history)
		self.history = []
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
		return 'Random'


"""
Manual strategies for testing the StrategyDetector
"""
class PlayerStrategy:
	"""
	Abstract class for a player strategy
	"""
	def __init__(self, name):
		self.name = name
		self.history = []

	def choose_action(self, opponent):
		pass

	def clear_history(self):
		self.history = []


class TitForTat(PlayerStrategy):
	def __init__(self):
		super().__init__("TitForTat")

	def choose_action(self, opponent):
		if len(opponent.history) == 0:
			self.history.append("C")
			return "C"
		else:
			self.history.append(opponent.history[-1])
			return opponent.history[-1]


class Dove(PlayerStrategy):
	def __init__(self):
		super().__init__("Dove")

	def choose_action(self, opponent):
		self.history.append("C")
		return "C"


class Hawk(PlayerStrategy):
	def __init__(self):
		super().__init__("Hawk")

	def choose_action(self, opponent):
		self.history.append("D")
		return "D"


class Random(PlayerStrategy):
	def __init__(self):
		super().__init__("Random")

	def choose_action(self, opponent):
		self.history.append(np.random.choice(["C", "D"]))
		return self.history[-1]


if __name__ == "__main__":
	possible_strats = [TitForTat(), Dove(), Hawk(), Random()]
	detector = StrategyDetector([c.name for c in possible_strats])

	for strat in possible_strats:
		verdict = detector.detect(player=strat)
		print(f"Player {strat.name} is seen as: {verdict}")
