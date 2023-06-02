import numpy as np
from constants import MAX_MEMORY_CAPACITY
import random

class Player:
    # Allows to have players with different memory capacities
    max_memory_capacity = MAX_MEMORY_CAPACITY
    def __init__(self, identifier:int, n_matchups:int, n_games:int, memory_capacity:int):
        """
        Args:
            identifier: unique identifier for player
            n_matchups: number of matchups per generation
            n_games: number of games per matchup
            memory_capacity: (optional) number of previous actions to consider when making a decision
        """
        self.identifier = identifier
        # Has to be smaller than max memory capacity
        self.memory_capacity = memory_capacity if memory_capacity <= Player.max_memory_capacity else Player.max_memory_capacity
        self.brain = MLP(n_input=memory_capacity, n_hidden=4)
        self.n_matchups = n_matchups
        self.n_games = n_games
        self.action_history = -np.ones(shape=(n_matchups, n_games + Player.max_memory_capacity), dtype=int)
        self.reward = 0
        self.n_matchups_played = 0
        self.initialize_action_history()

    @classmethod
    def t4tplayer(cls, identifier: int, n_matchups: int, n_games: int, memory_capacity=3, use_cuda=False):
        """hardcode a t4t player with memory 3. args are the same as __init__()

        Args:
            identifier: unique identifier for player
            n_matchups: number of matchups per generation
            n_games: number of games per matchup
            memory_capacity: (optional) number of previous actions to consider when making a decision
        """
        Player = cls(identifier, n_matchups, n_games, memory_capacity, use_cuda)
        # capacity = 3
        # Player.brain.W1_ = np.array([[-0.03709999634875267, -1.7961602824118976, -1.8527891849240585, -0.36712396749863263], [1.0231250035826696, -1.9575437119800085, 0.19572761446875336, 2.240618403104051], [3.1467075480031617, 2.268071929050388, -2.070629723532281, 1.2688080835096178]])
        # Player.brain.W2_ = np.array([[4.053400565815565], [1.8971396987502256], [-0.7128677902049138], [-0.869760297130533]])
        # Player.brain.Wb1 = np.array([[0.20788300846405733, -1.102598967259388, -0.4842573079389737, 0.9118712326838753]])
        # Player.brain.Wb2 = np.array([[-3.9004539936852285]])

        # capacity = 1
        Player.brain.W1_ = np.array([[-3.3558280112521226, 1.3545721381177211, 0.08247490964425366, 0.2547275421203078]])
        Player.brain.W2_ = np.array([[-1.6306096105362322], [3.8493623418727454], [0.2581273899162175], [2.6713216828002047]])
        Player.brain.Wb1 = np.array([[2.6942157638088875, 0.576930698019904, -2.1350991199552376, 0.05426302095116809]])
        Player.brain.Wb2 = np.array([[-0.2280021211335938]])
        
        return Player

    
    def initialize_action_history(self)->None:
        """Initializes first max_memory_capacity actions for all matchups with zeros."""
        init_actions = np.zeros(shape=(self.action_history.shape[0], Player.max_memory_capacity))
        self.action_history[:,:Player.max_memory_capacity] = init_actions
        
    def act(self, history: np.ndarray) -> np.ndarray: 
        """
        Args:
            history: array of observed actions
            
        Returns:
            actions: array of 0s and 1s, corresponding to cooperate or defect
        """
        actions = self.brain.forward(history)
        return actions
    
    def reset(self):
        """Reset action and reward history of player"""
        self.action_history = -np.ones(shape=(self.n_matchups, self.n_games + Player.max_memory_capacity), dtype=int)
        self.reward = 0
        self.n_matchups_played = 0
        self.initialize_action_history()

    def mutate(self):
        """Mutate neuron connections of brain"""

        loc = 0
        scale = 0.5
        self.brain.W1 += np.random.normal(loc, scale, size=self.brain.W1.shape)
        self.brain.W2 += np.random.normal(loc, scale, size=self.brain.W2.shape)
        self.brain.Wb1 += np.random.normal(loc, scale, size=self.brain.Wb1.shape)
        self.brain.Wb2 += np.random.normal(loc, scale, size=self.brain.Wb2.shape)


    def crossover(self, other):
        """Cross over genes of player with another

        Args:
            other (Player): player to cross over with
        """
        crossover_p = 0.1
        # switch weight vectors randomly between 2 players

        for i, col in enumerate(self.brain.W1[0]):
            if random.random() < crossover_p:
                temp = self.brain.W1[:,i]
                self.brain.W1[:,i] = other.brain.W1[:,i]
                other.brain.W1[:,i] = temp
        for i, col in enumerate(self.brain.W2[0]):
            if random.random() < crossover_p:
                temp = self.brain.W2[:,i]
                self.brain.W2[:,i] = other.brain.W2[:,i]
                other.brain.W2[:,i] = temp
        for i, col in enumerate(self.brain.Wb1[0]):
            if random.random() < crossover_p:
                temp = self.brain.Wb1[:,i]
                self.brain.Wb1[:,i] = other.brain.Wb1[:,i]
                other.brain.Wb1[:,i] = temp
        for i, col in enumerate(self.brain.Wb2[0]):
            if random.random() < crossover_p:
                temp = self.brain.Wb2[:,i]
                self.brain.Wb2[:,i] = other.brain.Wb2[:,i]
                other.brain.Wb2[:,i] = temp

                
        # for i, row in enumerate(self.brain.W2_)
        # ...

class MLP:
    """Creates a multi-layer perceptron with a single hidden layer."""
    def __init__(self, n_input, n_hidden):
        """
        Args:
            n_input: number of actions observed by player
            n_hidden: number of hidden units in the hidden layer
            bias: (optional) whether to include bias in linear layers
        """
        self.W1 = np.random.normal(loc=0, scale=1, size=(n_input, n_hidden))
        self.W2 = np.random.normal(loc=0, scale=1, size=(n_hidden, 1))
        self.Wb1 = np.random.normal(loc=0, scale=1, size=(1, n_hidden))
        self.Wb2 = np.random.normal(loc=0, scale=1, size=(1, 1))
        self.f1 = lambda x: np.maximum(0, x)  

    def forward(self, X: np.ndarray) -> np.ndarray:
        """ Args:
            X: input matrix of shape (n, m), where n is the number of opponents 
               and m is the number of actions observed 

            Returns:
            output: output matrix of shape (n, 1), one action for each opponent
            """ 
        output = self.f1(X @ self.W1 + self.Wb1) @ self.W2 + self.Wb2
        output = np.array(output >= 0, dtype=bool) * 1
        output = np.reshape(output, (output.shape[0],))
        return output
    
    class RMLP:
        """Creates a recurrent multi-layer perceptron with a single hidden layer."""
        def __init__(self, n_input, n_hidden):
            """
            Args:
                n_input: number of actions observed by player
                n_hidden: number of hidden units in the hidden layer
                bias: (optional) whether to include bias in linear layers
            """
            self.W1 = np.random.normal(loc=0, scale=2, size=(n_input, n_hidden))
            self.W2 = np.random.normal(loc=0, scale=2, size=(n_hidden, 1))
            self.Wh = np.random.normal(loc=0, scale=2, size=(n_hidden, n_hidden))
            self.Wb1 = np.random.normal(loc=0, scale=2, size=(1, n_hidden))
            self.Wb2 = np.random.normal(loc=0, scale=2, size=(1, 1))
            self.h = np.random.normal(loc=0, scale=2, size=(n_hidden, 2))
            self.f1 = lambda x: np.maximum(0, x)  

        def forward(self, X: np.ndarray) -> np.ndarray:
            """ Args:
                X: input matrix of shape (n, m), where n is the number of opponents 
                   and m is the number of actions observed 

                Returns:
                output: output matrix of shape (n, 1), one action for each opponent
                """ 
            self.h[:,1] = self.f1(X @ self.W1 + self.h[:,0] @ self.Wh + self.Wb1)
            output = self.h[:,1] @ self.W2 + self.Wb2
            output = np.array(output >= 0, dtype=bool) * 1
            output = np.reshape(output, (output.shape[0],))
            self.h[:,0] = self.h[:,1]
            return output

          