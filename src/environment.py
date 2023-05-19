import numpy as np
from player import Player
import time
import random

class PrisonersDilemma:

    """ Creates a Prisoner's Dilemma game."""

    def __init__(self, payoff_matrix=np.array([[(3,3), (0,5)], [(5,0), (1,1)]]), actions=['C', 'D']):

        """ parameters:
            payoff_matrix: 2x2 matrix of tuples, where each tuple is the payoff for each player
            actions: list of actions, where each action is a string
            """
        self.payoff_matrix = payoff_matrix
        self.actions = actions
              
    def to_action(self, p_history=list[int])-> list[str]:

        """ parameters:
            p_history: list of actions taken by player
            
            returns:
            history: list of actions taken by player, where each action is a string
            """ 
        history = p_history.copy()
        for i, a in enumerate(history):          
            history[i] = self.actions[history[i]]        
        return history


class Environment:

    """ Creates an environment for players to interact in."""

    def __init__(self, n_players:Player, n_games:int, n_matchups:int, game=PrisonersDilemma(), fitness=lambda x: np.sum(x)):

        """ parameters:
            players: list of players
            player_histories: list of player histories
            game: game to simulate
            fitness: fitness function for players
            """ 
        self.players = np.array([Player(identifier=i, n_matchups=n_matchups, n_games=n_games) for i in range(n_players)])
        self.n_matchups = n_matchups
        self.n_games = n_games
        self.game = game
        self.fitness = fitness 

    def run(self, n_generations:int)->None:

        """ parameters:
            n_generations: number of generations to simulate

            returns:
            players: list of surving players after n_generations
            """
        for _ in range(n_generations):
            # start_time = time.time()
            matchups = self.sample_matchups()
            for p in self.players:
                opponent_ids = matchups[matchups[:,p.identifier]>=0, p.identifier]
                if len(opponent_ids) == 0:
                    continue              
                self.simulate_game(player=p, opponent_ids=opponent_ids)
            return
            # end_time = time.time()
            # execution_time = end_time - start_time
            # print("Execution time:", execution_time, "seconds")
            self.evolve()
    
    def evolve(self)->None:
        # TODO: implement evolution algorithm
        # TODO: reset player rewards
        # TODO: vectorize computations
        pass
        
    # def sample_opponents(self, player:Player, n_matchups:int)-> list[Player]:

    #     """ parameters:
    #         player: player to sample opponents for
    #         n_matchups: number of opponents to sample

    #         returns:
    #         opponents: list of opponent ids
    #         """
    #     opponent_ids = []
    #     n_matchups_played = len(player.opponents) 
    #     n_matchups_remain = n_matchups - n_matchups_played
    #     while len(opponent_ids) < n_matchups_remain:
    #         opponent_id = np.random.randint(len(self.players))
    #         if len(self.players[opponent_id].opponents) == n_matchups:
    #             continue
    #         elif len(self.players[opponent_id].opponents) == n_matchups:
    #             continue
    #         opponent_ids.append(opponent_id)
    #         self.players[opponent_id].opponents.append(player.identifier)
    #         if opponent_id == player.identifier:
    #             continue
    #         player.opponents.append(opponent_id)
    #     return opponent_ids
    
    def sample_matchups(self) -> np.ndarray:
        # Create container for matchup info: 
        n_players = len(self.players)
        # rows = matchups, columns = player ids. -1 means no matchup
        matchups = -np.ones(shape=(self.n_matchups, n_players), dtype=int)
        for i in range(self.n_matchups):
            # Create set of players. Useful cause we can remove elements with O(1)
            player_set = set(range(n_players))
            for _ in range(n_players//2):
                # Random sample of 2 players. Also O(1)
                matchup = random.sample(player_set, 2)
                # Place matchup in container at two positions. 
                # Neglect reverse matchup. Prioritize lower id, makes forward passes more efficient 
                matchups[i,np.min(matchup)] = np.max(matchup)
                # Remove players from set to ensure same number of game for each player
                player_set.remove(matchup[0])
                player_set.remove(matchup[1])
        return matchups

    def simulate_game(self, player:Player, opponent_ids:np.ndarray) -> None: 
        n = len(opponent_ids)
        nth_player_matchup = player.matchups_played
        opponent_actions = np.empty(shape=(n, self.n_games+Player.max_memory_capacity), dtype=int)
        # extract matchup count for opponents
        opponent_matchups_played = np.array([oppo])
        for game_i in range(self.n_games):

            for i, id in enumerate(opponent_ids): 
                # Opponent's opponent is current player
                opponent = self.players[id]
                # Opponent's action is based on last opponent.memory_capacity moves of current player. 
                # If no games are played yet, opponent's action is random. Reflected by + player.memory_capacity
                upper = Player.max_memory_capacity + game_i
                lower = upper - opponent.memory_capacity
                action = opponent.act(player.action_history[i+nth_player_matchup-1,lower:upper])
                # Add action to opponent's action history
                nth_matchup = opponent_matchups_played[id]
                opponent.action_history[nth_matchup, upper] = action
                # Add action to opponent_actions. (Not the best way to do this)
                opponent_actions[i,:] = opponent.action_history[nth_matchup,:]
                # Increment number of matchups for oppoenents. 
                opponent_matchups_played[id] += 1
            # reset matchup count for opponents

            # Determine player actions based on opponent actions
            upper = Player.max_memory_capacity + game_i
            lower = upper - player.memory_capacity         
            player_actions = player.act(opponent_actions[:,lower:upper])
            # Add player actions to player action history
            player.action_history[nth_player_matchup:nth_player_matchup+n, upper] = player_actions
        # Increment number of matchups of players in matchup
        player.matchups_played += n


            # player_actions = player.act(opponent_histories).reshape(-1, 1)
            # if game_i == 0:
            #     player.history = p_actions
            # else:
            #     player.history = np.hstack((player.history, p_actions))
            # rewards = self.game.payoff_matrix[player.history[:,-1].reshape(n,),
            #                                   opponent_actions]
            # player.reward_history += rewards[:,0].tolist()
            # for i, id in enumerate(opponent_ids):
            #     opponent = self.players[id]
            #     if player.identifier == id:
            #         continue
            #     opponent.reward_history.append(rewards[i,1])



