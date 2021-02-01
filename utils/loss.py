import numpy as np

def softmax_ce_naive_forward_backward(X, W, y, reg):
    """Implémentation naive qui calcule la propagation avant, puis la
       propagation arrière pour finalement retourner la perte entropie croisée
       (ce) + une régularisation L2 et le gradient des poids. Utilise une 
       activation softmax en sortie.
       
       NOTE : la fonction codée est : EntropieCroisée + 0.5*reg*||W||^2
       
       N'oubliez pas le 0.5!

    Inputs:
    - X: Numpy array, shape (N, D). N représente le nombre d'exemple d'entrainement
        dans X, et D représente la dimension des exemples de X.
    - W: Numpy array, shape (D, C)
    - y: Numpy array, shape (N,). y[i] = c veut dire que X[i] appartient à la
        classe c, 0 <= c < C
    - reg: float. Terme de regularisation L2

    Outputs:
    - loss: float. Perte du classifieur linéaire softmax
    - dW: Numpy array, shape (D, C). Gradients des poids W
    """

    N = X.shape[0]
    C = W.shape[1]

    loss = 0
    dW = np.zeros(W.shape)

    ### TODO ###
    # Ajouter code ici #

    return loss, dW


def softmax_ce_forward_backward(X, W, y, reg):
    """Implémentation vectorisée qui calcule la propagation avant, puis la
       propagation arrière pour finalement retourner la perte entropie croisée
       (ce) et le gradient des poids. Utilise une activation softmax en sortie.
        
       NOTE : la fonction codée est : EntropieCroisée + 0.5*reg*||W||^2      
       N'oubliez pas le 0.5!

    Inputs:
    - X: Numpy array, shape (N, D). N représente le nombre d'exemples d'entrainement
        dans X, et D représente la dimension des exemples de X.
    - W: Numpy array, shape (D, C)
    - y: Numpy array, shape (N,). y[i] = c veut dire que X[i] appartient à la
        classe c, 0 <= c < C
    - reg: float. Terme de regularisation L2

    Outputs:
    - loss: float. Perte du classifieur linéaire softmax
    - dW: Numpy array, shape (D, C). Gradients des poids W
    """
    N = X.shape[0]
    C = W.shape[1]
    loss = 0.0
    dW = np.zeros(W.shape)
    ### TODO ###
    # Ajouter code ici #
    
    return loss, dW


def hinge_naive_forward_backward(X, W, y, reg):
    """Implémentation naive calculant la propagation avant, puis la
       propagation arrière, pour finalement retourner la perte hinge et le
       gradient des poids.
       
       NOTE : la fonction codée est : Hinge + 0.5*reg*||W||^2
       N'oubliez pas le 0.5!

    Inputs:
    - X: Numpy array, shape (N, D)
    - W: Numpy array, shape (D, C)
    - y: Numpy array, shape (N,). y[i] = c veut dire que X[i] appartient à la
         classe c, 0 <= c < C
    - reg: float. Terme de regularisation L2

    Outputs:
    - loss: float. Perte du classifieur linéaire hinge
    - dW: Numpy array, shape (D, C). Gradients des poids W
    """
    loss = 0.0
    dW = np.zeros(W.shape)

    ### TODO ###
    # We save the number of data points
    n = X.shape[0]

    for i in range(n):

        # We compute the predictions for each class (1 x C Numpy array)
        preds = np.dot(X[i:i+1, :], W)

        # We save the class with the highest score and the ground truth
        arg_max = np.argmax(preds)
        ground_truth = y[i]

        # We compute the loss
        loss += max(0, 1+preds[0, arg_max]-preds[0, ground_truth])

        # We update the gradients (Columns of W in this case)
        if arg_max != ground_truth:
            dW[:, arg_max:arg_max+1] += X[i:i+1, :].T
            dW[:, ground_truth:ground_truth+1] -= X[i:i+1, :].T

    # We take the mean per observation and add regularization (Squared of Frobenius Norm)
    loss = loss/n + 0.5*reg*pow(np.linalg.norm(W), 2)

    # We take the mean of the gradient and add the gradients with respect to the regularization
    dW = dW/n + reg*W

    return loss, dW


def hinge_forward_backward(X, W, y, reg):
    """Implémentation vectorisée calculant la propagation avant, puis la
       propagation arrière, pour finalement retourner la perte hinge et le
       gradient des poids.

       NOTE : la fonction codée est : Hinge + 0.5*reg*||W||^2
       N'oubliez pas le 0.5!
       
    Inputs:
    - X: Numpy array, shape (N, D)
    - W: Numpy array, shape (D, C)
    - y: Numpy array, shape (N,). y[i] = c veut dire que X[i] appartient à la
         classe c, 0 <= c < C
    - reg: float. Terme de regularisation L2

    Outputs:
    - loss: float. Perte du classifieur linéaire hinge
    - dW: Numpy array, shape (D, C). Gradients des poids W
    """
    loss = 0.0
    dW = np.zeros(W.shape)

    ### TODO ###
    # We save the number of classes
    C = W.shape[1]

    # We compute the predictions for each class (N x C Numpy array)
    preds = X.dot(W)

    # We stack one hot encodings for arg_max and ground_truth and take their differences
    arg_max = np.eye(C)[np.argmax(preds, axis=1)]   # (N x C Numpy array)
    ground_truth = np.eye(C)[y]                     # (N x C Numpy array)
    one_hot_diff = arg_max - ground_truth           # (N x C Numpy array)

    # We compute gradients and add regularization term
    dW = X.T.dot(one_hot_diff)/X.shape[0] + reg*W    # (D x C Numpy array)

    # We compute the differences between max prediction and ground truth prediction for all elements
    preds_diff = (preds * one_hot_diff).sum(axis=1)  # (N x 1 Numpy array)

    # We compute the mean of the loss
    loss = np.maximum(0, 1+preds_diff).mean() + 0.5*reg*pow(np.linalg.norm(W), 2)

    return loss, dW
