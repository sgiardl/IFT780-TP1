import numpy as np


class LinearClassifier:

    def __init__(self, forward_backward_function):
        self.W = None
        self.forward_backward_function = forward_backward_function

    def train(self, X, y, learning_rate=1e-3, weigth_scale=1e-3, reg=1e-5, num_iter=100, batch_size=200, verbose=False):
        """Boucle d'entraînement pour un classifieur linéaire.

        Arguments:
            X {ndarray} -- Données d'entraînement. Shape (N, D)
            y {ndarray} -- Labels d'entraînement. Shape (N, ).
                           Valeur y[i]: 0 <= y[i] < C, où C = nombre de classes

        Keyword Arguments:
            learning_rate {float} -- Taux d'apprentissage. (default: {1e-3})
            weigth_scale {float} -- Écart type de la distribution normale avec laquelle les poids
                                    sont initialisés. (default: {1e-3})
            reg {float} -- terme de régularisation. (default: {1e-5})
            num_iter {int} -- Nombre d'itérations de la boucle d'entraînement (default: {100})
            batch_size {int} -- Nombre de données d'entraînement passées au modèle par itération.
                                (default: {200})
            verbose {bool} -- (default: {False})

        Returns:
            list -- Historique de la perte lors de l'entraînement.
        """

        N, D = X.shape
        C = np.max(y) + 1

        if self.W is None:
            self.W = np.random.randn(D, C) * weigth_scale

        train_loss_history = []
        
        ### TODO ###
        for i in range(num_iter):

            # We select random indexes
            idx = np.random.choice(N, batch_size, replace=False)

            # We execute the forward and backward pass
            l, dW = self.forward_backward_function(X[idx, :], self.W, y[idx], reg)

            # We update the weights and add the loss to train_loss_history
            self.W -= learning_rate*dW
            train_loss_history.append(l)

            if verbose:
                print(f"Epoch {i} - Loss : {round(l,5)}")

        return train_loss_history

    def predict(self, X):
        scores = X.dot(self.W)
        return np.argmax(scores, axis=1)

    def forward_backward(self, X, y, reg):
        loss, dW = self.forward_backward_function(X, self.W, y, reg)
        return loss, dW
