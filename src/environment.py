import numpy as np
from player import Player
import random
import copy
import matplotlib.pyplot as plt
import multiprocessing as mp

class Environment:

    def __init__(self, n_players:int, n_matchups:int, n_games:int, fitness=lambda x,t:np.power(0.99,t) * np.sum(x)):
        """ Args:
            n_players: number of players
            n_matchups: number of matchups per generation
            n_games: number of games per matchup
            fitness: fitness function
            """
        self.payoff_matrix = np.array([[(3,3), (0,5)], [(5,0), (1,1)]])
        self.n_matchups = n_matchups
        self.n_games = n_games
        self.players = [Player(identifier=i, n_matchups=n_matchups, n_games=n_games) for i in range(n_players)]
        self.fitness = fitness 

    def run(self, n_generations:int, verbose=False)->None:
        """ Args:
            n_generations: number of generations to run
            
            Returns:
            None
            """
        for gen_i in range(n_generations):

            if verbose:
                print(f'----------------------------\nGENERATION {gen_i+1}')

            matchups = self.sample_matchups()
            for p in self.players:
                # Ignore entries with -1, which means no matchup
                opponent_ids = matchups[matchups[:,p.identifier]>=0, p.identifier]
                if len(opponent_ids) == 0:
                    continue              
                self.simulate_game(player=p, opponent_ids=opponent_ids, verbose=verbose)
            self.evolve()

    def evolve(self)->None:
        """Evolve generation of players by selecting fittest individuals and 
           generating their mutated mutated offspring."""
        # Percentage of players that are kept for next generation: Elite
        elite = 0.5
        index = int(elite * len(self.players))
        # Sort players in descending order. Elite is until index
        # Is it possible to integrate the player.reset() call in here? 
        self.players.sort(key=lambda x: x.reward, reverse=True)
        # Ids of players that do not belong to elite. To be given to new players in next generation
        available_ids = []
        for i, p in enumerate(self.players[index:]):
            available_ids.append(p.identifier)
        # Generate new players using surviving parents
        new_players = []
        # Create new players until all ids are used
        for id in available_ids:
            # Create new player
            child = Player(identifier=id, n_matchups=self.n_matchups, n_games=self.n_games)
            # Select random parent to inherit brain
            parent_index = random.randint(0, index-1)
            child.brain = copy.deepcopy(self.players[parent_index].brain)
            # Mutate brain of child
            child.mutate()
            # Add child
            new_players.append(child)
        # Create new generation
        self.players[index:] = new_players
        # Reset player information and ids
        for i, p in enumerate(self.players):
            p.reset()
            p.identifier = i
    
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

    def simulate_game(self, player:Player, opponent_ids:np.ndarray, verbose:bool) -> None: 
        """ Simulates a game between provided player and provided opponents from opponent ids.
        
            Args:
            player: The current player
            opponent_ids: Array of opponent ids
            verbose: If True, prints information about the game
            
            Returns:
            None
            """
        # Number of opponents
        n = len(opponent_ids)
        # Maximum memory capacity of the class player
        max_memory_capacity = Player.max_memory_capacity
        # Number of matchups played by player so far
        nth_player_matchup = player.n_matchups_played
        # Create container for opponent actions
        opponent_actions = -np.ones(shape=(n, self.n_games + max_memory_capacity), dtype=int)
        # Matchups played by opponents so far. Exclusively used for resetting
        n_matchups_played = np.array(list(map(lambda id: self.players[id].n_matchups_played, opponent_ids)))
        # Simulate games
        
        if verbose:
            print(f'----------------------------\nPlayer {player.identifier} vs. Players:{opponent_ids}')
        
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
                history = player.action_history[i + nth_player_matchup - 1, lower:upper]
                action = opponent.act(history)
                
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
            history = opponent_actions[:,lower:upper]        
            player_actions = player.act(history)

            if verbose:
                print(f'>> Player\'s actions: {player_actions}, actions observed: {np.ravel(history)}')
            
            # Add player actions to player action history
            player.action_history[nth_player_matchup:nth_player_matchup + n, upper] = player_actions
            # Determine rewards for player and opponents
            nth_player_matchup = player.n_matchups_played
            rewards = self.payoff_matrix[player.action_history[nth_player_matchup,upper], opponent_actions[:,upper]] 

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


    def plot_fitness(self, player_fitnesses):
        maxs = []
        mins = []
        avgs = []
        for generation in player_fitnesses:
            maxs.append(max(generation))
            mins.append(min(generation))
            avgs.append(sum(generation)/len(generation))
        plt.plot(maxs)
        plt.plot(avgs)
        plt.plot(mins)
        plt.xlabel('generation')
        plt.ylabel('fitness')
        plt.legend(['max fitness', 'avg fitness', 'min fitness'])
        plt.show()