import numpy as np
import math
from utils.activations import get_activation


class Dense:
    def __init__(self, dim_input=3*32*32, dim_output=10, weight_scale=1e-4, activation='identity'):
        """
        Keyword Arguments:
            dim_input {int} -- dimension du input de la couche. (default: {3*32*32})
            dim_output {int} -- nombre de neurones de notre couche (default: {10})
            weight_scale {float} -- écart type de la distribution normale utilisée
                                    pour l'initialisation des poids. Si None,
                                    initialisation Xavier ou He. (default: {1e-4})
            activation {str} -- identifiant de la fonction d'activation de la couche
                                (default: {'identity'})
        """
        
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.weight_scale = weight_scale
        self.activation_id = activation
        
        if weight_scale is not None:
            # Initialisation avec une distribution normale avec écart type = weight_scale
            self.W = np.random.normal(loc=0.0, scale=weight_scale, size=(dim_input, dim_output))
        elif activation == 'relu':
            # Initialisation 'He' avec une distribution normale
            self.W = np.random.normal(loc=0.0, scale=math.sqrt(2.0/dim_input), size=(dim_input, dim_output))
        else:
            # Initialisation 'Xavier' avec une distribution normale
            self.W = np.random.normal(loc=0.0, scale=math.sqrt(2.0/(dim_input + dim_output)),
                                      size=(dim_input, dim_output))

        self.b = np.zeros(dim_output)

        self.dW = 0
        self.db = 0
        self.reg = 0.0
        self.cache = None
        
        self.activation = get_activation(activation)
        
    def forward(self, X, **kwargs):
        """Effectue la propagation avant.  Le code de cette fonction est vectorisé.

        Arguments:
            X {ndarray} -- Input de la couche. Shape (N, dim_input)

        Returns:
            ndarray -- Sortie de la couche
        """
        self.cache = None
        
        # TODO

        # We calculate forward pass
        H = X.dot(self.W) + self.b
        A = self.activation['forward'](H)

        # We update the cache
        self.cache = {'H': H, 'A': A, 'X': X}

        return A

    def backward(self, dA, **kwargs):
        """Effectue la rétro-propagation pour les paramètres de la
           couche.

        Arguments:
            dA {ndarray} -- Dérivée de la loss par rapport à la sortie de la couche.
                            forme: (N, dim_output)

        Returns:
            ndarray -- Dérivée de la loss par rapport à l'entrée de la couche.
        """
        # TODO

        # We compute gradient of the loss according to W
        dA_dH = self.activation['backward'](self.cache['H'])  # (NxOUT numpy array)
        dH_dW = self.cache['X']                               # (NxIN numpy array)
        dL_dH = dA * dA_dH                                    # (NxOUT numpy array)
        self.dW = dH_dW.T.dot(dL_dH)                          # (InxOUT numpy array)

        # We compute the gradient of the loss according to the bias
        self.db = np.ones((1, self.dim_output))

        # Retourne la derivee de la couche courante par rapport à son entrée * la backProb dA
        return dA.dot(self.W.T)

    def get_params(self):
        return {'W': self.W, 'b': self.b}
        
    def get_gradients(self):
        return {'W': self.dW, 'b': self.db}

    def reset(self):
        self.__init__(dim_input=self.dim_input, 
                      dim_output=self.dim_output,
                      weight_scale=self.weight_scale, 
                      activation=self.activation_id)
