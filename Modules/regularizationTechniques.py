import numpy as np

class WeightDecay:
    def __init__(self, lam):
        """
        Initialize the L2 weight decay regularization module.

        Args:
            lam (float): Regularization strength (lambda).
        """

        self.lam = lam

    def compute_penalty(self, params):
        """
        Compute the L2 regularization penalty.

        Args:
            params (dict): Dictionary of model parameters.
                           Only keys starting with 'W' (weights) are considered.
        """

        penalty = 0
        for k in params:
            if k.startswith("W"):
                penalty += 0.5 * self.lam * np.sum(params[k] ** 2)
        return penalty

    def compute_gradients(self, params):
        """
        Compute the gradient of the L2 penalty with respect to weights.

        Args:
            params (dict): Dictionary of model parameters.
                           Only keys starting with 'W' are considered.
        """

        grads = {}
        for k in params:
            if k.startswith("W"):
                grads[k] = self.lam * params[k]
        return grads


class Dropout:
    def __init__(self, rate):
        """
        Initialize the Dropout layer.

        Args:
            rate (float): Dropout rate (fraction of units to drop). Should be between 0 and 1.
        """

        self.rate = rate
        self.mask = None

    def forward(self, x, training=True):
        """
        Apply dropout to the input during training.

        Args:
            x (ndarray): Input data.
            training (bool): Whether the model is in training mode.
        """

        if training and self.rate > 0:
            self.mask = np.random.binomial(1, 1 - self.rate, size=x.shape) / (1 - self.rate)
            return x * self.mask
        return x

    def backward(self, dout):
        """
        Backward pass through the dropout mask.

        Args:
            dout (ndarray): Gradient from the next layer.
        """

        return dout * self.mask if self.mask is not None else dout
