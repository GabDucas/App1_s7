import numpy as np

from dnn_framework.loss import Loss


class CrossEntropyLoss(Loss):
    """
    This class combines a softmax activation function and a cross entropy loss.
    """

    def calculate(self, x, target):
        """
        :param x: The input tensor (shape: (N, C))
        :param target: The target classes (shape: (N,))
        :return A tuple containing the loss and the gradient with respect to the input (loss, input_grad)
        """
        # shape retourne la size en x de la matrice
        N = x.shape[0]

        # Softmax est le 'sigmoid' de la classification multi-classes
        x = softmax(x)
        loss_matrix = -np.log(x[np.arange(N), target])
        loss = np.mean(loss_matrix)

        grad = x.copy()
        for i in range(N):
            grad[i, target[i]] -= 1
        grad /= N

        return loss, grad


def softmax(x):
    """
    Sert à sortir un vecteur de probabilité
    :param x: The input tensor (shape: (N, C))
    :return The softmax of x
    """
    row, column = x.shape
    out = np.zeros_like(x)

    for i in range(row):
        m = max(x[i])
        exp = np.exp(x[i] - m)
        s = sum(exp)
        out[i] = exp / s

    return out


class MeanSquaredErrorLoss(Loss):
    """
    This class implements a mean squared error loss.
    """

    def calculate(self, x, target):
        """
        :param x: The input tensor (shape: any)
        :param target: The target tensor (shape: same as x)
        :return A tuple containing the loss and the gradient with respect to the input (loss, input_grad)
        """
        N = x.shape[0] * x.shape[1]
        loss = np.mean((x - target) ** 2)

        grad = (2/N) * (x - target)
        return loss, grad