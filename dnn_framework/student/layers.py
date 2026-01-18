import numpy as np
from fontTools.misc.bezierTools import epsilon

from dnn_framework.layer import Layer


class FullyConnectedLayer(Layer):
    """
    This class implements a fully connected layer.
    """
    # Layer.weights = None
    # Layer.bias = None
    # Layer.cache = None # cache[0]=0

    def __init__(self, input_count, output_count):
        self.weights = np.random.randn(output_count, input_count) * 0.1
        self.bias = np.zeros(output_count)

    def get_parameters(self):
        # mutable (python shares references)
        return {
            'w': self.weights,
            'b': self.bias
        }

    def get_buffers(self):
        # new copy (will not change with training)
        return {
            'w': self.weights.copy(),
            'b': self.bias.copy()
        }

    def forward(self, x):
        y = x@self.weights.T + self.bias
        cache = x
        return y, cache

    def backward(self, output_grad, cache):
        x = cache
        dl_dw = output_grad.T @ x
        dl_db = np.sum(output_grad, axis=0)
        dl_dx = output_grad @ self.weights
        parameters_grad = {
            'w': dl_dw,
            'b': dl_db
        }
        return dl_dx, parameters_grad


class BatchNormalization(Layer):
    """
    This class implements a batch normalization layer.
    """

    def __init__(self, input_count, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        self.beta = np.zeros(input_count)
        self.gamma = np.ones(input_count)
        self.mean_t = np.zeros(input_count)
        self.var_t = np.zeros(input_count)

    def get_parameters(self):
        return {
            'beta': self.beta,
            'gamma': self.gamma
        }

    def get_buffers(self):
        return {
            'global_variance': self.var_t,
            'global_mean': self.mean_t,
        }

    def forward(self, x):
        if self.is_training():
            y,cache = self._forward_training(x)
        else:
            y,cache = self._forward_evaluation(x)
        return y, cache

    def _forward_training(self, x):
        #batch_size = x.shape[0]
        #mean = 1/batch_size * x
        #var = 1/batch_size * (x-mean)**2
        mean = np.mean(x, axis=0)
        var = np.var(x, axis=0)
        x_norm = (x - mean) / np.sqrt(var + epsilon)
        y = self.gamma * x_norm + self.beta
        cache = x, x_norm, mean, var

        self.mean_t = (1-self.alpha)*self.mean_t + self.alpha*mean
        self.var_t = (1-self.alpha)*self.var_t + self.alpha*var
        return y, cache

    def _forward_evaluation(self, x):
        x_norm = (x - self.mean_t) / np.sqrt(self.var_t + epsilon)
        y = self.gamma * x_norm + self.beta

        cache = x, x_norm, self.mean_t, self.var_t
        return y, cache

    def backward(self, output_grad, cache):
        x, x_norm, mean, var = cache
        batch_size = x.shape[0]
        dl_dx_norm = output_grad * self.gamma
        dl_dvar = np.sum(dl_dx_norm * (x-mean) * -0.5 * (var+epsilon)**-1.5, axis=0)
        dl_dmean = -np.sum(dl_dx_norm / np.sqrt(var + epsilon), axis=0)
        dl_dx = dl_dx_norm / np.sqrt(var + epsilon) + 2/batch_size * dl_dvar * (x-mean) + 1/batch_size * dl_dmean
        dl_dgamma = np.sum(output_grad * x_norm, axis=0)
        dl_dbeta = np.sum(output_grad, axis=0)

        parameters_grad = {
            'beta': dl_dbeta,
            'gamma': dl_dgamma
        }
        return dl_dx, parameters_grad


class Sigmoid(Layer):
    """
    This class implements a sigmoid activation function.
    """

    def get_parameters(self):
        return {}

    def get_buffers(self):
        return {}

    def forward(self, x):
        y = 1 / (1 + np.exp(-x))
        cache = y
        return y, cache

    def backward(self, output_grad, cache):
        y = cache
        dl_dx = (1 - y) * y * output_grad
        return dl_dx, {}


class ReLU(Layer):
    """
    This class implements a ReLU activation function.
    """

    def get_parameters(self):
        return {}

    def get_buffers(self):
        return {}

    def forward(self, x):
        y = np.copy(x)
        y[x < 0] = 0
        cache = x
        return y, cache

    def backward(self, output_grad, cache):
        x = cache
        dl_dx = np.copy(output_grad)
        dl_dx[x < 0] = 0
        return dl_dx, {}
