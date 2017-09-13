import numpy as np # Matrix and vector computation package
import matplotlib.pyplot as plt  # Plotting library
# Set the seed of the numpy random number generator so that the tutorial is reproducable
np.random.seed(seed=1)
from matplotlib.colors import colorConverter, ListedColormap # some plotting functions
import itertools
import collections

from nn_utils import logistic, logistic_deriv, softmax

# Define the layers used in this model
class Layer(object):
    """Base class for the different layers.
    Defines base methods and documentation of methods."""
    def __init__(self):
        self._name = None
        self._is_fc_layer = None
    
    def get_params_iter(self):
        """Return an iterator over the parameters (if any).
        The iterator has the same order as get_params_grad.
        The elements returned by the iterator are editable in-place."""
        return []
    
    def get_params_grad(self, X, output_grad):
        """Return a list of gradients over the parameters.
        The list has the same order as the get_params_iter iterator.
        X is the input.
        output_grad is the gradient at the output of this layer.
        """
        return []
    
    def get_output(self, X):
        """Perform the forward step linear transformation.
        X is the input."""
        pass
    
    def get_input_grad(self, Y, output_grad=None, T=None):
        """Return the gradient at the inputs of this layer.
        Y is the pre-computed output of this layer (not needed in this case).
        output_grad is the gradient at the output of this layer 
         (gradient at input of next layer).
        Output layer uses targets T to compute the gradient based on the 
         output error instead of output_grad"""
        pass

    @property
    def get_name(self):
        return self._name

    @property
    def is_fc_layer(self):
        return self._is_fc_layer


class LinearLayer(Layer):
    """The linear layer performs a linear transformation to its input."""
    
    def __init__(self, n_in, n_out, init_mode="normalize"):
        """Initialize hidden layer parameters.
        n_in is the number of input variables.
        n_out is the number of output variables."""
        if init_mode == "normalize":
            self.W = np.random.randn(n_in, n_out) * 0.1
        elif init_mode == "default":
            self.W = np.zeros((n_in, n_out))
        self.b = np.zeros(n_out)
        self._name = "fully_connected_layer"
        self.recv_buf = np.zeros((n_in+1, n_out))
        self._is_fc_layer = True
        
    def get_params_iter(self):
        """Return an iterator over the parameters."""
        return itertools.chain(np.nditer(self.W, op_flags=['readwrite']),
                               np.nditer(self.b, op_flags=['readwrite']))
    
    def get_output(self, X):
        """Perform the forward step linear transformation."""
        return X.dot(self.W) + self.b
        
    def get_params_grad(self, X, output_grad):
        """Return a list of gradients over the parameters."""
        JW = X.T.dot(output_grad)
        Jb = np.sum(output_grad, axis=0)
        return [g for g in itertools.chain(np.nditer(JW), np.nditer(Jb))]
        #return np.array([g for g in itertools.chain(np.nditer(JW), np.nditer(Jb))])
    
    def get_input_grad(self, Y, output_grad):
        """Return the gradient at the inputs of this layer."""
        return output_grad.dot(np.transpose(self.W))

    @property
    def get_shape(self):
        return self.W.shape

    @property
    def fetch_wrapped_layer(self):
        """
        combine W and b together for communication
        say previously W \in \mathbb{R}^{m, n}; b \in \mathbb{R} ^ n
        after combination, the shape should be \in \mathbb{R}^{m+1, n}
        """
        tmp_b = self.b.reshape(1, self.b.shape[0])
        return np.concatenate((self.W, tmp_b), axis=0)


class SigmoidLayer(Layer):
    """The logistic layer applies the logistic function to its inputs."""
    def __init__(self):
        self._name = 'sigmoid_layer'
        self._is_fc_layer = False
    
    def get_output(self, X):
        """Perform the forward step transformation."""
        return logistic(X)
    
    def get_input_grad(self, Y, output_grad):
        """Return the gradient at the inputs of this layer."""
        return np.multiply(logistic_deriv(Y), output_grad)


class SoftmaxOutputLayer(Layer):
    """The softmax output layer computes the classification propabilities at the output."""

    def __init__(self):
        self._name = 'softmax_layer'
        self._is_fc_layer = False
    
    def get_output(self, X):
        """Perform the forward step transformation."""
        return softmax(X)
    
    def get_input_grad(self, Y, T):
        """Return the gradient at the inputs of this layer."""
        return (Y - T) / Y.shape[0]
    
    def get_cost(self, Y, T):
        """Return the cost at the output of this output layer."""
        return - np.multiply(T, np.log(Y)).sum() / Y.shape[0]