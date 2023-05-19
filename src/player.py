import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Player:
    """Creates player with a unique identifier and a reasoning mechanism."""
    max_memory_capacity = 3
    def __init__(self, identifier:int, n_matchups:int, n_games:int, memory_capacity=3):
        """ parameters:
            identifier: unique identifier for player
            memory_capacity: number of previous actions to consider when making a decision
            """ 
        self.identifier = identifier
        self.memory_capacity = memory_capacity
        self.brain = MLP(n_input=memory_capacity, n_hidden=4).to(device)
        self.action_history = np.empty(shape=(n_matchups, n_games+Player.max_memory_capacity), dtype=int)
        self.reward_history = np.empty(shape=n_games, dtype=int)
        self.matchups_played = 0
        self.initialize_action_history()
    
    def initialize_action_history(self)->None:
        """Initializes action history with random actions."""
        random_actions = np.random.randint(2, size=(self.action_history.shape[0], Player.max_memory_capacity))
        self.action_history[:,:Player.max_memory_capacity] = random_actions
        
    def act(self, history: np.ndarray) -> np.ndarray: 
        """ parameters:
            history: list of observed actions
            
            returns:
            actions: array of 0s and 1s, corresponding to cooperate or defect
            """       
        actions = self.brain.forward_non_cuda(history)
        return actions


class MLP(nn.Module):

    """Creates a multi-layer perceptron with a single hidden layer."""

    def __init__(self, n_input, n_hidden, bias=True):
        """ parameters:
            n_input: number of actions observed by player
            n_hidden: number of hidden units in the hidden layer 
            activation: activation function for hidden layer (default: ReLU)

            attributes:
            W1: weight matrix for input to hidden layer
            W2: weight matrix for hidden layer to output
            Wb1: bias matrix for input to hidden layer
            Wb2: bias matrix for hidden layer to output
            f1: activation function for hidden layer

            returns:
            None 
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
        """ parameters:
            X: input matrix of shape (n, m), where n is the number of opponents 
               and m is the number of actions observed 

            returns:
            output: output matrix of shape (n, 1), one action for each opponent
            """ 
        X = np.array(X)
        if len(X.shape) == 1:
            X = np.reshape(X, (1, X.shape[0]))
        n = X.shape[1]
        if n < self.n_input:
            X = np.hstack((np.random.randint(2, size=(X.shape[0], self.n_input-n)), X))
        else:
            X = X[:, -self.n_input:]
        X = torch.from_numpy(X).float().to(device)
        Y = self.W1(X)
        Y = self.relu(self.W2(Y))
        out = Y.detach().cpu().numpy()
        out = np.array(out >= 0, dtype=np.int32).reshape(out.shape[0],)
        return out

    def forward_non_cuda(self, X: np.ndarray)->np.ndarray:
        """Forward pass not using cuda"""
        output = self.f1(X @ self.W1_ + self.Wb1) @ self.W2_ + self.Wb2
        output = np.array(output >= 0, dtype=bool) * 1
        output = np.reshape(output, (output.shape[0],))
        return output

          