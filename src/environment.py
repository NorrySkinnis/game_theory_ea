import numpy as np
from player import Player
import time
from tqdm import tqdm

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
        self.game = game
        self.fitness = fitness 

    def run(self, n_games:int, n_matchups:int, n_generations:int)->None:

        """ parameters:
            n_games: number of games to simulate
            n_matchups: number of opponents to sample
            n_generations: number of generations to simulate

            returns:
            players: list of surving players after n_generations
            """
        for _ in tqdm(range(n_generations), desc="Generations"):
            #start_time = time.time()
            for p in self.players:
                opponent_ids = self.sample_opponents(p, n_matchups)
                if len(opponent_ids) == 0:
                    continue
                self.simulate_game(p, opponent_ids, n_games)
                # self.simulate_game_2(p, opponent_ids, n_games)
            # end_time = time.time()
            # execution_time = end_time - start_time
            # print("Execution time:", execution_time, "seconds")
            self.players = self.evolve()

    
    def evolve(self)->None:
        # TODO: implement evolution algorithm
        # TODO: reset player rewards
        # TODO: vectorize computations
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
            elif len(self.players[opponent_id].opponents) == n_matchups:
                continue
            opponent_ids.append(opponent_id)
            self.players[opponent_id].opponents.append(player.identifier)
            player.opponents.append(opponent_id)
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
        for game_i in tqdm(range(n_games), desc="Games"):
            opponent_actions = []
            for i, id in enumerate(opponent_ids): 
                opponent = self.players[id]
                o_action = opponent.act(player.history[i])
                opponent_actions.append(o_action[0])
                opponent_histories[i].append(o_action[0])
            p_actions = player.act(opponent_histories).reshape(-1, 1)
            if game_i == 0:
                player.history = p_actions
            else:
                player.history = np.hstack((player.history, p_actions))
            rewards = self.game.payoff_matrix[player.history[:,-1].reshape(n,),
                                              opponent_actions]
            p_rewards = np.sum(rewards[:,0])
            player.rewards += p_rewards
            opponent_rewards = rewards[:,1]
            for i, id in enumerate(opponent_ids):
                opponent = self.players[id]
                opponent.rewards += opponent_rewards[i]

    # Same functionality, twice as slow
    # def simulate_game_2(self, player:Player, opponent_ids:list[int], n_games:int) -> None:

    #     n = len(opponent_ids)
    #     opponent_histories = [[] for _ in range(n)]
    #     player.history = [[] for _ in range(n)]
    #     for _ in range(n_games):
    #         for i, id in enumerate(opponent_ids): 
    #             opponent = self.players[id]
    #             o_action = opponent.act(player.history[i])
    #             opponent_histories[i].append(o_action[0])
    #             p_action = player.act(opponent_histories[i])
    #             player.history[i].append(p_action[0])
    #             rewards = self.game.payoff_matrix[p_action[0], o_action[0]]
    #             player.rewards += rewards[0]
    #             opponent.rewards += rewards[1]

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




