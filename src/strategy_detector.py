# Generic imports
import numpy as np
from itertools import product

# Custom imports
from player import Player
from constants import STRATEGY_CODES, MAX_MEMORY_CAPACITY

class StrategyDetector:
	"""Creates a detector that can determine the current player's strategy.
	
	Given the memory capacity of the current player, the detector constructs an array
	containg all possible inputs. 
	
	Using these inputs, player actions are derived for each of them.
	This will create a unique mapping that is then transformed into a unique code.
	
	This code can be linked to a strategy.
	
	Example: 
	--------
	Memory capacity = 1: 0 -> C, 1 -> D

	Possible inputs: [0], [1]
	
	Detector strategy: [[0], [1]]
	
	Possible player codes: [0, 0], [0, 1], [1, 0], [1, 1]
	
	Transformed player codes: {}, {1}, {0}, {0, 1}
	"""
	def __init__(self):
		self.player_strategy_code = None
		self.strategy = self.set_strategy()

	def set_strategy(self)->dict[int, np.ndarray]:
		"""Constructs a dictionary of all possible input combinations for each memory capacity.
		
		Returns:
		--------
		strategy: (dict[int, np.ndarray])
			Dictionary of all possible input combinations for each memory capacity.
		"""
		keys = [c for c in range(1, MAX_MEMORY_CAPACITY+1)]
		permutations = {key: None for key in keys}
		for c in range(1,MAX_MEMORY_CAPACITY+1):
			permutations[c] = np.array(list(product([0,1], repeat=c)))
		return permutations

	def detect_strategy(self, player:Player, verbose:bool)->int:
		"""Detector plays all possible input combination for the current player.
		
		Constructs unique code from the inputs for which the player defected.
		Denoted by the index of the input combination in the strategy dictionary.
			
		Parameters:
		----------
		player: (Player)
			Player whose strategy is to be detected.	
		
		verbose: (bool)
			Flag to print strategy detection process.
		"""
		# Get player's memory capacity
		memory_capacity = player.memory_capacity
		# Get detector strategy for player's memory capacity
		strategy = self.strategy[memory_capacity]
		# initialize the strategy code for player
		player_code = set()
		# Play all possible input combinations
		for i, history in enumerate(strategy):
			player_action = player.act(history)

			if verbose:
				print(f'Input: {history} -> Action: {player_action}')

			# If player defected, add index of input comination to player_code
			if player_action == 1:
				player_code.add(i)

		if verbose:
			print(f'Player code: {player_code}')

		# Save player_code
		self.player_strategy_code = player_code
		# Determine player strategy from player code
		player_strategy = self.strategy_from_code(memory_capacity=memory_capacity)
		return player_strategy

	def strategy_from_code(self, memory_capacity:int)->int:
		"""Determines the player's strategy from the player code.
		
		Parameters:
		----------
		memory_capacity: (int)
			Memory capacity of the player.

		Returns:
		--------
		strategy_id: (int)
			Id linked to strategy of the player.
		"""
		# Get the strategy codes for the given memory capacity
		strategy_codes = STRATEGY_CODES[memory_capacity]
		# Loop over all of them and find the matching one
		for strategy_id, code in enumerate(strategy_codes):
			if self.player_strategy_code == code:
				return strategy_id
		return 5