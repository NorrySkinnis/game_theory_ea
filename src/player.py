# Generic imports
import numpy as np
import random

# Custom imports
from constants import MAX_MEMORY_CAPACITY

class Player:
    def __init__(self, identifier:int, n_matchups:int, n_games:int, memory_capacity:int):
        self.identifier = identifier
        self.n_matchups = n_matchups
        self.n_games = n_games
        self.memory_capacity = memory_capacity 

        # Initialize brain, action history, reward, and number of matchups played
        self.brain = MLP(n_input=memory_capacity, n_hidden=4)
        # self.brain = RMLP(n_input=memory_capacity, n_hidden=4, n_matchups=n_matchups)
        self.action_history = -np.ones(shape=(n_matchups, n_games+MAX_MEMORY_CAPACITY), dtype=int)
        self.reward = 0
        self.n_matchups_played = 0
        self.initialize_action_history()
    
    def initialize_action_history(self)->None:
        """ Initializes first max_memory_capacity actions for all matchups with zeros.
        
        Ensures that the brain gets an input before the first game is played.
        """
        init_actions = np.zeros(shape=(self.action_history.shape[0], MAX_MEMORY_CAPACITY), dtype=int)
        self.action_history[:,:MAX_MEMORY_CAPACITY] = init_actions
        
    def act(self, history:np.ndarray) -> np.ndarray: 
        """ Produces action based on observed history
        Parameters:
        -----------
        history: (np.ndarray)
            Array of actions observed by player.

        Returns:
        --------
        action: (np.ndarray)
            Array of actions. 0 = cooperate, 1 = defect.
        """
        actions = self.brain.forward(history)
        return actions
    
    def reset(self):
        """Resets action and reward history of player."""
        self.action_history = -np.ones(shape=(self.n_matchups, self.n_games + MAX_MEMORY_CAPACITY), dtype=int)
        self.reward = 0
        self.n_matchups_played = 0
        self.initialize_action_history()

        
    ###### MEMORY CAPACITY 3 ######
    @classmethod
    def t4tplayer(cls, identifier: int, n_matchups: int, n_games: int, memory_capacity=3):
        """hardcode a t4t player with memory 3. args are the same as __init__()

        Args:
            identifier: unique identifier for player
            n_matchups: number of matchups per generation
            n_games: number of games per matchup
            memory_capacity: (optional) number of previous actions to consider when making a decision
        """
        Player = cls(identifier, n_matchups, n_games, memory_capacity)
        Player.brain.W1 = np.array([[-0.0371, -1.79616028, -1.85278918, -0.36712397], 
                                    [ 1.023125, -1.95754371,  0.19572761,  2.2406184 ], 
                                    [ 3.14670755 , 2.26807193, -2.07062972,  1.26880808]])
        Player.brain.W2 = np.array([[ 4.05340057], [ 1.8971397 ], [-0.71286779], [-0.8697603 ]])
        Player.brain.Wb1 = np.array([[ 0.20788301, -1.10259897, -0.48425731,  0.91187123]])
        Player.brain.Wb2 = np.array([[-3.90045399]])
        return Player

class MLP():
    """Creates a multi-layer perceptron with a single hidden layer."""
    def __init__(self, n_input, n_hidden):
        # Weights of first fully connected layer
        self.W1 = np.random.normal(loc=0, scale=1, size=(n_input, n_hidden))
        # Weights of second fully connected layer
        self.W2 = np.random.normal(loc=0, scale=1, size=(n_hidden, 1))
        # Bias of first fully connected layer
        self.Wb1 = np.random.normal(loc=0, scale=1, size=(1, n_hidden))
        # Bias of second fully connected layer
        self.Wb2 = np.random.normal(loc=0, scale=1, size=(1, 1))
        # Activation function (ReLU)
        self.f1 = lambda x: np.maximum(0, x)  

    def forward(self, X: np.ndarray) -> np.ndarray:
        """ Forwards input through network and returns actions.

        Parameters:
        -----------
        X: (np.ndarray)
            Input matrix of shape (n, m), where n is the number of opponents 
            and m is the number of actions observed.
        
        Returns:
        --------
        output: (np.ndarray)
            Output matrix of shape (n, 1), one action for each opponent. 
        """
        output = self.f1(X @ self.W1 + self.Wb1) @ self.W2 + self.Wb2
        output = np.array(output >= 0, dtype=bool) * 1
        output = np.reshape(output, (output.shape[0],))
        return output

    def mutate(self, mutation_rate:float):
        """Mutate neuron connections of brain.

        Parameters:
        -----------
        mutation_rate: (float)
            Probability of mutation.
        """
        loc = 0
        scale = 1
        self.W1 += np.random.normal(loc, scale, size=self.W1.shape) * \
            np.random.choice([True, False], size=self.W1.shape, p=[mutation_rate, 1-mutation_rate])
        self.W2 += np.random.normal(loc, scale, size=self.W2.shape) * \
            np.random.choice([True, False], size=self.W2.shape, p=[mutation_rate, 1-mutation_rate])
        self.Wb1 += np.random.normal(loc, scale, size=self.Wb1.shape) * \
            np.random.choice([True, False], size=self.Wb1.shape, p=[mutation_rate, 1-mutation_rate])
        self.Wb2 += np.random.normal(loc, scale, size=self.Wb2.shape) * \
            np.random.choice([True, False], size=self.Wb2.shape, p=[mutation_rate, 1-mutation_rate])

    def crossover(self, other, crossover_p:float):
        """Cross over genes of player with another

        Args:
            other (Player): player to cross over with
        """
        for i in range(self.W1.shape[1]):
            if random.random() < crossover_p:
                temp_w = self.W1[:,i]
                temp_b = self.Wb1[:,i]
                self.W1[:,i] = other.W1[:,i]
                self.Wb1[:,i] = other.Wb1[:,i]
                other.W1[:,i] = temp_w
                other.Wb1[:,i] = temp_b
    
class RMLP():
    """Creates a recurrent multi-layer perceptron with a single hidden layer."""
    def __init__(self, n_input:int, n_hidden:int, n_matchups:int):
        # Weights of first fully connected layer
        self.W1 = np.random.normal(loc=0, scale=1, size=(n_input, n_hidden))
        # Weights of second fully connected layer
        self.W2 = np.random.normal(loc=0, scale=1, size=(n_hidden, 1))
        # Weights of hidden layer
        self.Wh = np.random.normal(loc=0, scale=1, size=(n_hidden, n_hidden))
        # Bias of first fully connected layer
        self.Wb1 = np.random.normal(loc=0, scale=1, size=(1, n_hidden))
        # Bias of second fully connected layer
        self.Wb2 = np.random.normal(loc=0, scale=1, size=(1, 1))
        # Initialize data container for hidden states 
        self.h = np.random.normal(loc=0, scale=1, size=(n_matchups, n_hidden, 2))
        # Activation function (ReLU)
        self.f1 = lambda x: np.maximum(0, x)  

    def forward(self, X: np.ndarray) -> np.ndarray:
        """ Forwards input through network and returns actions.

        Parameters:
        -----------
        X: (np.ndarray)
            Input matrix of shape (n, m), where n is the number of opponents 
            and m is the number of actions observed.
        
        Returns:
        --------
        output: (np.ndarray)
            Output matrix of shape (n, 1), one action for each opponent. 
        """
        n_opponents = X.shape[0]
        # Compute hiddent state using previous hidden state
        self.h[:n_opponents,:,1] = self.f1(X @ self.W1 + self.h[:n_opponents,:,0] @ self.Wh + self.Wb1)
        output = self.h[:n_opponents,:,1] @ self.W2 + self.Wb2
        output = np.array(output >= 0, dtype=bool) * 1
        output = np.reshape(output, (output.shape[0],))
        # Update hidden state
        self.h[:n_opponents,:,0]= self.h[:n_opponents,:,1]

        return output
    
    def mutate(self, mutation_rate:float):
        """ Mutate neuron connections of brain.
        
        Parameters:
        -----------
        mutation_rate: (float)
            Probability of mutation.
        """
        # Strength of mutation
        loc = 0
        scale = 1

        self.W1 += np.random.normal(loc, scale, size=self.W1.shape) * \
            np.random.choice([True, False], size=self.W1.shape, p=[mutation_rate, 1-mutation_rate])
        self.W2 += np.random.normal(loc, scale, size=self.W2.shape) * \
            np.random.choice([True, False], size=self.W2.shape, p=[mutation_rate, 1-mutation_rate])
        self.Wh += np.random.normal(loc, scale, size=self.Wh.shape) * \
            np.random.choice([True, False], size=self.Wh.shape, p=[mutation_rate, 1-mutation_rate])
        self.Wb1 += np.random.normal(loc, scale, size=self.Wb1.shape) * \
            np.random.choice([True, False], size=self.Wb1.shape, p=[mutation_rate, 1-mutation_rate])
        self.Wb2 += np.random.normal(loc, scale, size=self.Wb2.shape) * \
            np.random.choice([True, False], size=self.Wb2.shape, p=[mutation_rate, 1-mutation_rate])

          