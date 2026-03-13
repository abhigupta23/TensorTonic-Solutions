import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """

    X = np.array(X)
    y = np.array(y)
    D = X.shape[1]
    N = X.shape[0]
    
    W = np.zeros(D)
    # b = np.zeros(N)
    b = 0.0

    for i in range(steps):
        
        Z = (X @ W) + b
        P = _sigmoid(Z)
    
        grad_W = (X.T @ (P - y))/N
        grad_b = np.mean(P-y)
    
        W = W - (lr * grad_W)
        b = b - (lr * grad_b)
    
    return W,b