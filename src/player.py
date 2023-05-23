import numpy as np
import torch
import torch.nn as nn
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Player:
    # Allows to have players with different memory capacities
    max_memory_capacity = 1
    def __init__(self, identifier:int, n_matchups:int, n_games:int, memory_capacity=1):
        """ Args:
            identifier: unique identifier for player
            n_matchups: number of matchups per generation
            n_games: number of games per matchup
            memory_capacity: number of previous actions to consider when making a decision
            """ 
        self.identifier = identifier
        # Has to be smaller than max memory capacity
        self.memory_capacity = memory_capacity if memory_capacity <= Player.max_memory_capacity else Player.max_memory_capacity
        self.brain = MLP(n_input=memory_capacity, n_hidden=4).to(device)
        self.n_matchups = n_matchups
        self.n_games = n_games
        self.action_history = -np.ones(shape=(n_matchups, n_games + Player.max_memory_capacity), dtype=int)
        self.reward_history = -np.ones(shape=(n_matchups, n_games), dtype=int)
        self.matchups_played = 0
        self.initialize_action_history()
    
    def initialize_action_history(self)->None:
        """Initializes action history with random actions."""
        random_actions = np.random.randint(2, size=(self.action_history.shape[0], Player.max_memory_capacity))
        self.action_history[:,:Player.max_memory_capacity] = random_actions
        
    def act(self, history: np.ndarray) -> np.ndarray: 
        """ Args:
            history: array of observed actions
            
            Returns:
            actions: array of 0s and 1s, corresponding to cooperate or defect
            """       
        actions = self.brain.forward_non_cuda(history)
        return actions
    
    def __lt__(self, other):
        """Helper function to sort players by reward history

        Args:
            other (Player): Player to compare total reward to

        Returns:
            Bool: True if greater, False if smaller
        """
        # This won't work until reward history is fixed
        return np.sum(self.reward_history) > np.sum(other.reward_history)
    
    def reset_history(self):
        """Reset action and reward history of player
        """
        self.action_history = -np.ones(shape=(self.n_matchups, self.n_games + Player.max_memory_capacity), dtype=int)
        self.reward_history = -np.ones(shape=(self.n_matchups, self.n_games), dtype=int)
        self.matchups_played = 0
        self.initialize_action_history()

    def mutate(self):
        """Mutate brain by adding random noise
        """
        lower = -1
        upper = 1
        for w in self.brain.W1_:
            w += random.uniform(lower, upper)
        for w in self.brain.W2_:
            w += random.uniform(lower, upper)
        for w in self.brain.Wb1:
            w += random.uniform(lower, upper)
        for w in self.brain.Wb2:
            w += random.uniform(lower, upper)



class MLP(nn.Module):
    """Creates a multi-layer perceptron with a single hidden layer."""
    def __init__(self, n_input, n_hidden, bias=True):
        """ Args:
            n_input: number of actions observed by player
            n_hidden: number of hidden units in the hidden layer 
            activation: activation function for hidden layer (default: ReLU)
            """
        super(MLP, self).__init__()
        self.n_input = n_input
        self.W1 = nn.Linear(n_input, n_hidden, bias=bias)
        self.W2 = nn.Linear(n_hidden, 1, bias=bias)
        self.relu = nn.ReLU()

        self.W1_ = np.random.normal(loc=0, scale=2, size=(n_input, n_hidden))
        self.W2_ = np.random.normal(loc=0, scale=2, size=(n_hidden, 1))
        self.Wb1 = np.random.normal(loc=0, scale=2, size=(1, n_hidden))
        self.Wb2 = np.random.normal(loc=0, scale=2, size=(1, 1))
        self.f1 = lambda x: np.maximum(0, x)

    def forward(self, X: np.ndarray)-> np.ndarray:
        """ Args:
            X: input matrix of shape (n, m), where n is the number of opponents 
               and m is the number of actions observed 

            Returns:
            output: output matrix of shape (n, 1), one action for each opponent
            """ 
        X = torch.tensor(X, dtype=torch.float32).to(device)
        y = self.W1(X)
        y = self.relu(self.W2(y))
        out = y.detach().cpu().numpy()
        out = np.array(out >= 0, dtype=np.int32).reshape(out.shape[0],)
        return out

    def forward_non_cuda(self, X: np.ndarray)->np.ndarray:
        """Forward pass not using cuda"""
        output = self.f1(X @ self.W1_ + self.Wb1) @ self.W2_ + self.Wb2
        output = np.array(output >= 0, dtype=bool) * 1
        output = np.reshape(output, (output.shape[0],))
        return output

          