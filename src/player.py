import numpy as np

class Player:

    """Creates player with a unique identifier and a reasoning mechanism."""
    
    def __init__(self, identifier:int, memory_capacity=3):

        """ parameters:
            identifier: unique identifier for player
            memory_capacity: number of previous actions to consider when making a decision
            """
        self.identifier = identifier
        self.brain = MLP(n_input=memory_capacity, n_hidden=4)
        self.history = None
        self.reward_history = []
        self.opponents = []
        
    def act(self, history: list) -> np.ndarray[int]: 

        """ parameters: 
            history: list of observed actions
            
            returns:
            actions: array of 0s and 1s, corresponding to cooperate or defect
            """       
        actions = self.brain.forward(history)
        return actions
    
class MLP:

    """Creates a multi-layer perceptron with a single hidden layer."""

    def __init__(self, n_input, n_hidden, activation=lambda x: np.maximum(0,x)): 

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
        self.W1 = np.random.normal(loc=0, scale=2, size=(n_input, n_hidden))
        self.W2 = np.random.normal(loc=0, scale=2, size=(n_hidden, 1))
        self.Wb1 = np.random.normal(loc=0, scale=2, size=(1, n_hidden))
        self.Wb2 = np.random.normal(loc=0, scale=2, size=(1, 1))
        self.f1 = activation
    
    def forward(self, X: list)-> np.ndarray:
         
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
        m = self.W1.shape[0]
        if n < m:
            X = np.hstack((np.random.randint(2, size=(X.shape[0], m-n)), X))
        else:
            X = X[:,-m:]
        output = self.f1(X @ self.W1 + self.Wb1) @ self.W2 + self.Wb2
        output = np.array(output >= 0, dtype=bool) * 1
        output = np.reshape(output, (output.shape[0],))
        return output
          