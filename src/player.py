import numpy as np
import torch
import torch.nn as nn
from constants import device, MAX_MEMORY_CAPACITY
import random


class Player:
    # Allows to have players with different memory capacities
    max_memory_capacity = MAX_MEMORY_CAPACITY
    def __init__(self, identifier: int, n_matchups: int, n_games: int, memory_capacity=2, use_cuda=False):
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
        self.brain = MLP(n_input=memory_capacity, n_hidden=4).to(device)
        self.n_matchups = n_matchups
        self.n_games = n_games
        self.use_cuda = use_cuda
        self.action_history = -np.ones(shape=(n_matchups, n_games + Player.max_memory_capacity), dtype=int)
        self.reward = 0
        self.n_matchups_played = 0
        self.initialize_action_history()
    
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
        if self.use_cuda:
            actions = self.brain(history)
        else:
            actions = self.brain.forward_non_cuda(history)
        return actions
    
    def reset(self):
        """Reset action and reward history of player"""
        self.action_history = -np.ones(shape=(self.n_matchups, self.n_games + Player.max_memory_capacity), dtype=int)
        self.reward = 0
        self.n_matchups_played = 0
        self.initialize_action_history()

    def mutate(self):
        """Mutate neuron connections of brain"""
        self.brain.W1_ += np.random.normal(loc=0, scale=1, size=self.brain.W1_.shape)
        self.brain.W2_ += np.random.normal(loc=0, scale=1, size=self.brain.W2_.shape)
        self.brain.Wb1 += np.random.normal(loc=0, scale=1, size=self.brain.Wb1.shape)
        self.brain.Wb2 += np.random.normal(loc=0, scale=1, size=self.brain.Wb2.shape)

    def crossover(self, other):
        """Cross over genes of player with another

        Args:
            other (Player): player to cross over with
        """
        crossover_p = 0.1
        # switch weight vectors randomly between 2 players
        for i, row in enumerate(self.brain.W1_):
            if random.random() < crossover_p:
                temp = row
                row = other.brain.W1_[i]
                other.brain.W1_[i] = temp
                
        # for i, row in enumerate(self.brain.W2_)
        # ...



class MLP(nn.Module):
    """Creates a multi-layer perceptron with a single hidden layer."""
    def __init__(self, n_input, n_hidden, bias=True):
        """
        Args:
            n_input: number of actions observed by player
            n_hidden: number of hidden units in the hidden layer
            bias: (optional) whether to include bias in linear layers
        """
        super(MLP, self).__init__()
        self.W1 = nn.Linear(n_input, n_hidden, bias=bias)
        self.W2 = nn.Linear(n_hidden, 1, bias=bias)
        self.relu = nn.ReLU()

        self.W1_ = np.random.normal(loc=0, scale=2, size=(n_input, n_hidden))
        self.W2_ = np.random.normal(loc=0, scale=2, size=(n_hidden, 1))
        self.Wb1 = np.random.normal(loc=0, scale=2, size=(1, n_hidden))
        self.Wb2 = np.random.normal(loc=0, scale=2, size=(1, 1))
        self.f1 = lambda x: np.maximum(0, x)  # manual ReLU activation function

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
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

    def forward_non_cuda(self, X: np.ndarray) -> np.ndarray:
        """ Args:
            X: input matrix of shape (n, m), where n is the number of opponents 
               and m is the number of actions observed 

            Returns:
            output: output matrix of shape (n, 1), one action for each opponent
            """ 
        output = self.f1(X @ self.W1_ + self.Wb1) @ self.W2_ + self.Wb2
        output = np.array(output >= 0, dtype=bool) * 1
        output = np.reshape(output, (output.shape[0],))
        return output

          