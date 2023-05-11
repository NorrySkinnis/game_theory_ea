import numpy as np
from player import Player as player

class PrisonersDilemma:

    """ Creates a Prisoner's Dilemma game."""

    def __init__(self, payoff_matrix=[[(3,3), (0,5)], [(5,0), (1,1)]], actions=['C', 'D']):

        """ parameters:
            payoff_matrix: 2x2 matrix of tuples, where each tuple is the payoff for each player
            actions: list of actions, where each action is a string
            """
        self.payoff_matrix = payoff_matrix
        self.actions = actions
        
    def simulate(self, player:player, opponents:list[int], n_games:int) -> None: 
        """
        parameters:
        player: player to simulate games for
        opponents: list of possible opponents
        n_games: number of games to play
        
        returns: None
        """ 

        o_histories = []
        o_actions = []
        for game_i in range(n_games):
            for i, opponent in enumerate(opponents):    
                o_histories.append(opponent.history[0]) 
                o_action = opponent.act(player.history[i])
                opponent.history.append(o_action)
                o_actions.append(o_action)
            p_actions = player.act(o_histories)
            # cannot slice list, have to find better way to append player history 
            
            
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

    def __init__(self, players:list[player], game=PrisonersDilemma(), fitness=lambda x: np.sum(x)):

        """ parameters:
            players: list of players

        """   
        self.players = players
        self.game = game
        self.fitness = fitness 
    
    def evolve(self, n_games, n_matchups, n_generations):     
        for p in self.players:
            opponents = self.sample_opponents(p, n_matchups)
            self.game.simulate(p, opponents, n_games)
        pass
    
    def sample_opponents(self, player:player, n_matchups:int)-> list[player]:
        """ parameters:
            player: player to sample opponents for
            n_matchups: number of opponents to sample

            returns:
            opponents: list of opponents
            """
        opponents = []
        n_matchups_played = len(player.get_opponents())
        n_matchups_remain = n_matchups - n_matchups_played
        while len(opponents) < n_matchups_remain:
            opponent_id = np.random.randint(len(self.players))
            if opponent_id == player.get_id():
                continue
            opponents.append(self.players[opponent_id])
        return opponents




