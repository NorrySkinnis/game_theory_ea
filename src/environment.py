import numpy as np
from player import Player

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

    def __init__(self, n_players=1000, game=PrisonersDilemma(), fitness=lambda x: np.sum(x)):

        """ parameters:
            players: list of players
            player_histories: list of player histories
            game: game to simulate
            fitness: fitness function for players
            """ 
        self.players = self.create_players(n_players)
        self.player_histories = [p.history for p in self.players]
        self.game = game
        self.fitness = fitness 

    def run(self, n_games:int, n_matchups:int, n_generations:int)->list[Player]:

        """ parameters:
            n_games: number of games to simulate
            n_matchups: number of opponents to sample
            n_generations: number of generations to simulate

            returns:
            players: list of surving players after n_generations
            """
        for g in range(n_generations):
            for p in self.players:
                opponent_ids = self.sample_opponents(p, n_matchups)
                self.simulate_game(p, opponent_ids, n_games)
                return
            self.evolve()
    
    def evolve(self)->None:
        # TODO: implement evolution algorithm
        pass
        
    def sample_opponents(self, player:Player, n_matchups:int)-> list[Player]:

        """ parameters:
            player: player to sample opponents for
            n_matchups: number of opponents to sample

            returns:
            opponents: list of opponent ids
            """
        opponent_ids = []
        n_matchups_played = len(player.opponents)
        n_matchups_remain = n_matchups - n_matchups_played
        while len(opponent_ids) < n_matchups_remain:
            opponent_id = np.random.randint(len(self.players))
            if opponent_id == player.identifier:
                continue
            elif self.players[opponent_id].opponents == n_matchups:
                continue
            opponent_ids.append(opponent_id)
        return opponent_ids
    
    def simulate_game(self, player:Player, opponent_ids:list[int], n_games:int) -> None: 

        """ parameters:
            player: player to simulate
            opponents: list of opponent ids
            n_games: number of games to simulate

            returns:
            None
            """
        n = len(opponent_ids)
        opponent_histories = [[] for _ in range(n)]  
        player.history = [[] for _ in range(n)] 
        for game_i in range(n_games):
            for i, id in enumerate(opponent_ids): 
                opponent = self.players[id]
                o_action = opponent.act(player.history[i])
                opponent_histories[i].append(o_action[0])
            p_actions = player.act(opponent_histories).reshape(-1,1)
            if game_i == 0:
                player.history = p_actions
            else:
                player.history = np.hstack((player.history, p_actions))
            rewards = self.game.payoff_matrix[(1,1), (1,1)]
            # print(np.ravel(opponent_histories))
            # print(player.history.reshape(1,-1).flatten())

            print(rewards[list(zip([1,0],[0,1]))])
            print(rewards[(1,1)])
            return
            
            



    def create_players(self, n_players:int)-> list[Player]:
        
        """ parameters:
            n_players: number of players to create

            returns:
            players: list of players
            """
        players = []
        for i in range(n_players):
            players.append(Player(i))
        return players




