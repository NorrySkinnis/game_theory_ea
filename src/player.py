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
        # capacity = 3
        # Player.brain.W1_ = np.array([[-0.03709999634875267, -1.7961602824118976, -1.8527891849240585, -0.36712396749863263], [1.0231250035826696, -1.9575437119800085, 0.19572761446875336, 2.240618403104051], [3.1467075480031617, 2.268071929050388, -2.070629723532281, 1.2688080835096178]])
        # Player.brain.W2_ = np.array([[4.053400565815565], [1.8971396987502256], [-0.7128677902049138], [-0.869760297130533]])
        # Player.brain.Wb1 = np.array([[0.20788300846405733, -1.102598967259388, -0.4842573079389737, 0.9118712326838753]])
        # Player.brain.Wb2 = np.array([[-3.9004539936852285]])

        # capacity = 1
        Player.brain.W1 = np.array([[0.39317879088646795, -1.1621958799752339, 1.5090907218007212, -0.578700719287135], [0.2663267808857011, -1.9243267343076034, -2.263988110778135, 3.2121356772178773]])
        Player.brain.W2 = np.array([[-0.2582632261183214], [-0.6516885143700015], [-0.0703517480993997], [1.5090354704682072]])
        Player.brain.Wb1 = np.array([[0.6608086435270056, 1.2096007035349565, 0.2686518690291412, 0.3545352896601341]])
        Player.brain.Wb2 = np.array([[-0.8375013776060737]])
        
        return Player

    # def crossover(self, other):
    #     """Cross over genes of player with another

    #     Args:
    #         other (Player): player to cross over with
    #     """
    #     crossover_p = 0.1
    #     # switch weight vectors randomly between 2 players

    #     for i, col in enumerate(self.brain.W1[0]):
    #         if random.random() < crossover_p:
    #             temp = self.brain.W1[:,i]
    #             self.brain.W1[:,i] = other.brain.W1[:,i]
    #             other.brain.W1[:,i] = temp
    #     for i, col in enumerate(self.brain.W2[0]):
    #         if random.random() < crossover_p:
    #             temp = self.brain.W2[:,i]
    #             self.brain.W2[:,i] = other.brain.W2[:,i]
    #             other.brain.W2[:,i] = temp
    #     for i, col in enumerate(self.brain.Wb1[0]):
    #         if random.random() < crossover_p:
    #             temp = self.brain.Wb1[:,i]
    #             self.brain.Wb1[:,i] = other.brain.Wb1[:,i]
    #             other.brain.Wb1[:,i] = temp
    #     for i, col in enumerate(self.brain.Wb2[0]):
    #         if random.random() < crossover_p:
    #             temp = self.brain.Wb2[:,i]
    #             self.brain.Wb2[:,i] = other.brain.Wb2[:,i]
    #             other.brain.Wb2[:,i] = temp
    
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

    def mutate(self):
        """Mutate neuron connections of brain."""
        loc = 0
        scale = 1
        self.W1 += np.random.normal(loc, scale, size=self.W1.shape)
        self.W2 += np.random.normal(loc, scale, size=self.W2.shape)
        self.Wb1 += np.random.normal(loc, scale, size=self.Wb1.shape)
        self.Wb2 += np.random.normal(loc, scale, size=self.Wb2.shape)
    
    class RMLP():
        """Creates a recurrent multi-layer perceptron with a single hidden layer."""
        def __init__(self, n_input, n_hidden):
            # Weights of first fully connected layer
            self.W1 = np.random.normal(loc=0, scale=2, size=(n_input, n_hidden))
            # Weights of second fully connected layer
            self.W2 = np.random.normal(loc=0, scale=2, size=(n_hidden, 1))
            # Weights of hidden layer
            self.Wh = np.random.normal(loc=0, scale=2, size=(n_hidden, n_hidden))
            # Bias of first fully connected layer
            self.Wb1 = np.random.normal(loc=0, scale=2, size=(1, n_hidden))
            # Bias of second fully connected layer
            self.Wb2 = np.random.normal(loc=0, scale=2, size=(1, 1))
            # Hidden state 
            self.h = np.random.normal(loc=0, scale=2, size=(n_hidden, 2))
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
            # Compute hiddent state using previous hidden state
            self.h[:,1] = self.f1(X @ self.W1 + self.h[:,0] @ self.Wh + self.Wb1)
            output = self.h[:,1] @ self.W2 + self.Wb2
            output = np.array(output >= 0, dtype=bool) * 1
            output = np.reshape(output, (output.shape[0],))
            # Update hidden state
            self.h[:,0] = self.h[:,1]
            return output
        
        def mutate(self):
            """ Mutate neuron connections of brain."""
            loc = 0
            scale = 1
            self.W1 += np.random.normal(loc, scale, size=self.W1.shape)
            self.W2 += np.random.normal(loc, scale, size=self.W2.shape)
            self.Wh += np.random.normal(loc, scale, size=self.Wh.shape)
            self.Wb1 += np.random.normal(loc, scale, size=self.Wb1.shape)
            self.Wb2 += np.random.normal(loc, scale, size=self.Wb2.shape)

          