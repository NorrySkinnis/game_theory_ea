# Generic imports
import numpy as np
import random
from tqdm import tqdm
import copy

# Custom imports
from player import Player
from evaluation import Evaluator
from strategy_detector import StrategyDetector
from constants import MAX_MEMORY_CAPACITY


class Environment:
    """Creates an environment for the IPD game."""

    def __init__(self, n_players: int, n_generations: int, n_matchups: int, n_games: int, memory_capacity: int, 
                 elite: float, mutation_rate: float, crossover_rate: float, payoff_matrix: np.ndarray, fitness: callable):
        self.n_generations = n_generations
        self.n_matchups = n_matchups
        self.n_games = n_games
        self.elite = elite
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.payoff_matrix = payoff_matrix
        self.fitness = fitness 
        self.players = self.initialize_players(n_players=n_players, memory_capacity=memory_capacity)
        self.detector = StrategyDetector()
    
    def initialize_players(self, n_players: int, memory_capacity: int) -> list[Player]:
        """Initializes the players of the environment.

        Parameters:
        -----------
        n_players: (int)
            Number of players in the population.
        
        memory_capacity: (int)
            Number of previous games to be remembered by the player.
        """
        return [Player(identifier=id, n_matchups=self.n_matchups, n_games=self.n_games, memory_capacity=memory_capacity) 
                for id in range(n_players)]
    
    def attach(self, evaluator: Evaluator) -> None:
        """Attaches an evaluator to the environment.

        Parameters:
        -----------
        evaluator: (Evaluator)
            Evaluator to be attached.
        """
        evaluator.nth_simulation += 1
        self.evaluator = evaluator

    def run(self, verbose=False) -> None:
        """ 
        Parameters:
        -----------
        verbose: (bool)
            If True, prints information about the game.
        """
        for nth_generation in tqdm(range(self.n_generations)):

            if verbose:
                print(f'----------------------------\nGENERATION {nth_generation+1}')

            matchups = self.sample_matchups()
            for p in self.players:
                # Ignore entries with -1, which means no matchup
                opponent_ids = matchups[matchups[:, p.identifier] >= 0, p.identifier]
                self.simulate_game(player=p, opponent_ids=opponent_ids, verbose=verbose)
                self.detector.detect_strategy(player=p, verbose=verbose)
                self.evaluator.update(player=p, nth_generation=nth_generation)           
            self.evolve()

    def evolve(self) -> None:
        """ Evolve generation of players by selecting the fittest individuals 
        and generating their mutated offspring.
        """
        # Percentage of players that are kept for next generation: Elite
        index = int(self.elite * len(self.players))
        # Sort players in descending order. Elite is until index.
        self.players.sort(key=lambda x: x.reward, reverse=True)
        # Ids of players that do not belong to elite. To be given to new players in next generation
        available_ids = []
        for p in self.players[index:]:
            available_ids.append(p.identifier)
        # Generate new players using surviving parents
        new_players = []
        # Create new players until all ids are used
        for id in available_ids:
            # Select two random parents from elite.
            parent_indeces = random.sample(set(range(index)), 2)
            parent1 = self.players[parent_indeces[0]]
            parent2 = self.players[parent_indeces[1]]
            # Create child and a dummy child to crossover with
            child = Player(identifier=id, n_matchups=self.n_matchups, n_games=self.n_games, 
                           memory_capacity=parent1.memory_capacity)
            dummy_child = Player(identifier=id, n_matchups=self.n_matchups, n_games=self.n_games, 
                                 memory_capacity=parent2.memory_capacity)  
            child.brain = copy.deepcopy(parent1.brain)
            dummy_child.brain = copy.deepcopy(parent2.brain)
            child.brain.crossover(other=dummy_child.brain, crossover_rate=self.crossover_rate)
            # Mutate brain of child
            child.brain.mutate(mutation_rate=self.mutation_rate)
            # One child policy. 
            new_players.append(child)
        # Create new generation
        self.players[index:] = new_players
        # Reset player information and ids. Ids HAVE to be sorted ascending in new list
        for id, p in enumerate(self.players):
            p.reset()
            p.identifier = id
    
    def sample_matchups(self) -> np.ndarray:
        """ Samples matchups for each player. Currently, with replacement."""
        n_players = len(self.players)
        # Create container for matchup information
        # rows = matchups, columns = player ids. -1 means no matchup
        matchups = -np.ones(shape=(self.n_matchups, n_players), dtype=int)
        for i in range(self.n_matchups):
            # Create set of players. Remove elements is O(1)
            player_set = set(range(n_players))
            for _ in range(n_players//2):
                # Random sample of 2 players. Also O(1)
                matchup = random.sample(player_set, 2)
                # Neglect reverse matchup. Prioritize lower id, makes forward passes more efficient 
                matchups[i, np.min(matchup)] = np.max(matchup)
                # Remove players from set to ensure same number of games for each player
                player_set.remove(matchup[0])
                player_set.remove(matchup[1])
        return matchups

    def simulate_game(self, player:Player, opponent_ids:np.ndarray, verbose:bool) -> None: 
        """
        Simulates a series of games between current player and opponents.
        
        Parameters:
        -----------
        player: (Player)
            Player to simulate games for.
        
        opponent_ids: (np.ndarray)
            Array of opponent ids.
        
        verbose: (bool)
            If True, prints information about the game.
        """
        # Number of opponents
        n = len(opponent_ids)
        # If no opponents, return
        if n == 0:
            return
        # Maximum memory capacity of the class player
        max_memory_capacity = MAX_MEMORY_CAPACITY
        # Number of matchups played by player so far
        nth_player_matchup = player.n_matchups_played
        # Create container for opponent actions
        opponent_actions = -np.ones(shape=(n, self.n_games + max_memory_capacity), dtype=int)
        # Matchups played by opponents so far. Exclusively used for resetting
        n_matchups_played = np.array(list(map(lambda id: self.players[id].n_matchups_played, opponent_ids)))
        
        if verbose:
            print(f'----------------------------\nPlayer {player.identifier} vs. Players:{opponent_ids}')
        
        # Simulate games
        for game_i in range(self.n_games): 
            
            if verbose:
                print(f'----------------------------\nGame {game_i+1}\n----------------------------')
            
            # Create set for unique opponent ids for each game
            unique_opponent_ids = set()
            for i, id in enumerate(opponent_ids): 
                # Current opponent
                opponent = self.players[id]
                # Reset opponent matchups played if not unique opponent.
                if id not in unique_opponent_ids:
                    # Reset matchups played by this opponent
                    opponent.n_matchups_played = n_matchups_played[i]
                    # Add opponent id to set if not already in set
                    unique_opponent_ids.add(id)            
                # Opponent's action is based on the last #opponent.memory_capacity moves of current player. 
                # If less games than opponent's memory capacity are played yet, then that opponent bases action on 
                # (memory capacity - # games played) random moves + (# games played) moves of current player 
                upper = max_memory_capacity + game_i
                lower = upper - opponent.memory_capacity
                history = player.action_history[i + nth_player_matchup, lower:upper]
                action = opponent.act(history.reshape(1,-1))
                
                if verbose:
                    print(f'Opponent {opponent.identifier}, action: {action}, actions observed: {history}')
                
                # Update match count for current opponent in matchups_played_updated
                nth_matchup = opponent.n_matchups_played
                # Add action to opponent's action history
                opponent.action_history[nth_matchup, upper] = action
                # Add action to opponent_actions
                opponent_actions[i,:] = opponent.action_history[nth_matchup,:]
                # Increment number of matchups for current opponent 
                opponent.n_matchups_played += 1
            # Determine player actions based on slice of all opponents' actions
            upper = max_memory_capacity + game_i
            lower = upper - player.memory_capacity 
            history = opponent_actions[:, lower:upper]
            player_actions = player.act(history)

            if verbose:
                print(f'>> Player\'s actions: {player_actions}, actions observed: {np.ravel(history)}')
            
            # Add player actions to player action history
            player.action_history[nth_player_matchup:nth_player_matchup + n, upper] = player_actions
            # nth_player_matchup = player.n_matchups_played
            # Transform actions to indeces
            indeces_player_actions = np.array(player_actions >= 0, dtype=bool) * 1
            indeces_opponent_actions = np.array(opponent_actions[:,upper] >= 0, dtype=bool) * 1
            rewards = self.payoff_matrix[indeces_player_actions, indeces_opponent_actions]
            # rewards = self.payoff_matrix[player_actions, opponent_actions[:,upper]]

            if verbose:
                print(f'>> Rewards: Opponents: {rewards[:,1]}, Player: {rewards[:,0]}')

            # Reward player
            player.reward += self.fitness(rewards[:,0], game_i)
            # Reward opponents
            for i, id in enumerate(opponent_ids):
                opponent = self.players[id]
                opponent.reward += self.fitness(rewards[i,1], game_i)
        # Update number of matchups played by player
        player.n_matchups_played += n