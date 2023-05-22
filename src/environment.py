import numpy as np
from player import Player
import random

class Environment:

    def __init__(self, n_players:int, n_matchups:int, n_games:int, fitness=lambda x: np.sum(x)):
        """ Args:
            n_players: number of players
            n_matchups: number of matchups per generation
            n_games: number of games per matchup
            fitness: fitness function for evolution algorithm
            """
        self.payoff_matrix = np.array([[(3,3), (0,5)], [(5,0), (1,1)]])
        self.players = np.array([Player(identifier=i, n_matchups=n_matchups, n_games=n_games) for i in range(n_players)])
        self.n_matchups = n_matchups
        self.n_games = n_games
        self.fitness = fitness 

    def run(self, n_generations:int)->None:
        """ Args:
            n_generations: number of generations to run
            
            Returns:
            None
            """
        for _ in range(n_generations):
            matchups = self.sample_matchups()
            for p in self.players:
                # Ignore entries with -1, which means no matchup
                opponent_ids = matchups[matchups[:,p.identifier]>=0, p.identifier]
                if len(opponent_ids) == 0:
                    continue              
                self.simulate_game(player=p, opponent_ids=opponent_ids)
            self.evolve()
    
    def evolve(self)->None:
        # TODO: implement evolution algorithm
        # TODO: reset player rewards
        # TODO: vectorize computations
        pass
    
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
                matchups[i,np.min(matchup)] = np.max(matchup)
                # Remove players from set to ensure same number of games for each player
                player_set.remove(matchup[0])
                player_set.remove(matchup[1])
        return matchups

    def simulate_game(self, player:Player, opponent_ids:np.ndarray) -> None: 
        """ Simulates a game between provided player and provided opponents from opponent ids.
        
            Args:
            player: The current player
            opponent_ids: Array of opponent ids
            
            Returns:
            None
            """
        # Number of opponents
        n = len(opponent_ids)
        # Maximum memory capacity of the class player
        max_memory_capacity = Player.max_memory_capacity
        # Number of matchups played by player so far
        nth_player_matchup = player.matchups_played
        # Create container for opponent actions
        opponent_actions = -np.ones(shape=(n, self.n_games + max_memory_capacity), dtype=int)
        # Matchups played by opponents so far. Exclusively used for resetting
        matchups_played = np.array(list(map(lambda id: self.players[id].matchups_played, opponent_ids)))
        # Matchups played by opponents so far. Used for updating.
        matchups_played_updated = matchups_played.copy()
        # Simulate games
        for game_i in range(self.n_games): 
            # Create set for unique opponent ids for each game
            unique_opponent_ids = set()
            for i, id in enumerate(opponent_ids): 
                # Current opponent
                opponent = self.players[id]
                # Reset opponent matchups played if not unique opponent.
                if id not in unique_opponent_ids:
                    # Reset matchups played by this opponent
                    opponent.matchups_played = matchups_played[i]
                    # Add opponent id to set if not already in set
                    unique_opponent_ids.add(id)            
                # Opponent's action is based on the last #opponent.memory_capacity moves of current player. 
                # If less games than opponent's memory capacity are played yet, then that opponent bases action on 
                # (memory capacity - # games played) random moves + (# games played) moves of current player 
                upper = max_memory_capacity + game_i
                lower = upper - opponent.memory_capacity
                action = opponent.act(player.action_history[i + nth_player_matchup - 1, lower:upper])
                nth_matchup = opponent.matchups_played
                # Update match count for current opponent in matchups_played_updated
                matchups_played_updated[i] = nth_matchup 
                # Add action to opponent's action history
                opponent.action_history[nth_matchup, upper] = action
                # Add action to opponent_actions
                opponent_actions[i,:] = opponent.action_history[nth_matchup,:]
                # Increment number of matchups for current opponent 
                opponent.matchups_played += 1
            # Determine player actions based on all opponents' actions
            upper = max_memory_capacity + game_i
            lower = upper - player.memory_capacity         
            player_actions = player.act(opponent_actions[:,lower:upper])
            # Add player actions to player action history
            player.action_history[nth_player_matchup:nth_player_matchup + n, upper] = player_actions
            # Determine rewards for player and opponents
            nth_player_matchup = player.matchups_played
            rewards = self.payoff_matrix[player.action_history[nth_player_matchup:,upper], opponent_actions[:,upper]] 
            # Add rewards to player's reward history
            player.reward_history[nth_player_matchup:, game_i] = rewards[:,0]
            # Add rewards to opponents' reward history
            for i, id in enumerate(opponent_ids):
                opponent = self.players[id]
                opponent.reward_history[matchups_played_updated[i]:, game_i] = rewards[i,1]  
        # Increment number of matchups of player
        player.matchups_played += n