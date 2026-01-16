import numpy as np

from dnn_framework.layer import Layer


class FullyConnectedLayer(Layer):
    """
    This class implements a fully connected layer.
    """
    # Layer.weights = None
    # Layer.bias = None
    # Layer.cache = None # cache[0]=0

    def __init__(self, input_count, output_count):
        self.weights = np.zeros((output_count, input_count)) # np.random.randn(output_count, input_count) * 0.01
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
        dl_dw = output_grad.T @ cache
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
        raise NotImplementedError()

    def get_parameters(self):
        raise NotImplementedError()

    def get_buffers(self):
        raise NotImplementedError()

    def forward(self, x):
        raise NotImplementedError()

    def _forward_training(self, x):
        raise NotImplementedError()

    def _forward_evaluation(self, x):
        raise NotImplementedError()

    def backward(self, output_grad, cache):
        raise NotImplementedError()


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
