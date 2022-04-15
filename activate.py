import numpy as np
import math

def sigmoid(a):
    return 1 / (1 + np.exp(-a))
    #derivative of sigmoid
def gradient_sigmoid(a):
        return sigmoid(a) * (1 - sigmoid(a))

def Softmax(a):
        e_x = np.exp(a - np.max(a, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)
def gradient_Softmax(a):
        p = Softmax(a)
        return p * (1 - p)

def TanH(a):
        return 2 / (1 + np.exp(-2*a)) - 1
def gradient_TanH(a):
    return 1 - np.power(TanH(a), 2)

def ReLU(a):
        return np.where(a >= 0, a, 0)
def gradient_ReLU(a):
    return np.where(a >= 0, 1, 0)       



def LeakyReLU(x, alpha=0.2):
    return np.where(x >= 0, x, alpha * x)
def gradient_LeakyReLU(x, alpha=0.2):
    return np.where(x >= 0, 1, alpha)


def ELU(x, alpha=0.1):
    return np.where(x >= 0.0, x, alpha * (np.exp(x) - 1))

def gradient_ELU(x, alpha=0.1):
    return np.where(x >= 0.0, 1, ELU(x) + alpha)

def SoftPlus(x):
    return np.log(1 + np.exp(x))

def gradient_SoftPlus(x):
    return 1 / (1 + np.exp(-x))

#Loss functions

# def CrossEntropy(y, p):
#         # Avoid division by zero
#     p = np.clip(p, 1e-15, 1 - 1e-15)
#     print(p)
#     return - y * np.log(p) - (1 - y) * np.log(1 - p)

# def gradient(y, p):
#     p = np.clip(p, 1e-15, 1 - 1e-15)
#     return - (y / p) + (1 - y) / (1 - p)
# h = np.array([0.10, 0.40, 0.50])
# q = np.array([0.80, 0.15, 0.05])
# hr=CrossEntropy(h, q)
#print(hr)
def _forward_pass(self, X, training=True):
        """ Calculate the output of the NN """
        layer_output = X
        for layer in self.layers:
            layer_output = layer.forward_pass(layer_output, training)

        return layer_output
def train_test_split(X, y, test_size=0.5, shuffle=True, seed=None):
    """ Split the data into train and test sets """
    if shuffle:
        X, y = shuffle_data(X, y, seed)
    # Split the training data from test data in the ratio specified in
    # test_size
    split_i = len(y) - int(len(y) // (1 / test_size))
    X_train, X_test = X[:split_i], X[split_i:]
    y_train, y_test = y[:split_i], y[split_i:]

    return X_train, X_test, y_train, y_test
def shuffle_data(X, y, seed=None):
    """ Random shuffle of the samples in X and y """
    if seed:
        np.random.seed(seed)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    return X[idx], y[idx]    
