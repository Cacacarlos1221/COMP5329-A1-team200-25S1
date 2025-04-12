import numpy as np

class MiniBatchGenerator:
    def __init__(self, batch_size=32):
        """
        Initialize the mini-batch generator with a specified batch size.
        
        Args:
            batch_size (int): Number of samples per mini-batch. Default is 32.
        """
        self.batch_size = batch_size

    def generate_batches(self, X, y):
        """
        Randomly shuffle the dataset and generate mini-batches.
        
        Args:
            X (ndarray): Feature matrix with shape (n_samples, n_features).
            y (ndarray): Label vector with shape (n_samples,).
        """
        n_samples = X.shape[0]
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        for start_idx in range(0, n_samples, self.batch_size):
            batch_idx = indices[start_idx:start_idx + self.batch_size]
            yield X[batch_idx], y[batch_idx]

class BatchNormalization:
    def __init__(self, dim, eps=1e-5, momentum=0.9):
        """
        Initialize the batch normalization layer.
        
        Args:
            dim (int): Number of features (input dimension).
            eps (float): Small constant to avoid division by zero.
            momentum (float): Momentum for updating running statistics.
        """
        
        self.eps = eps
        self.momentum = momentum
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        self.running_mean = np.zeros(dim)
        self.running_var = np.ones(dim)

    def forward(self, x, training=True):
        """
        Perform batch normalization on input data.
        
        Args:
            x (ndarray): Input data with shape (batch_size, dim).
            training (bool): Whether in training mode (True) or inference mode (False).
        """

        if training:
            mean = np.mean(x, axis=0)
            var = np.var(x, axis=0)
            self.x_centered = x - mean
            self.std_inv = 1. / np.sqrt(var + self.eps)
            self.x_norm = self.x_centered * self.std_inv
            out = self.gamma * self.x_norm + self.beta

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
            return out
        else:
            x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            return self.gamma * x_norm + self.beta

    def backward(self, dout):
        """
        Backpropagate through the batch normalization layer.
        
        Args:
            dout (ndarray): Gradient of the loss with respect to the output.
        """

        N = dout.shape[0]

        dbeta = np.sum(dout, axis=0)
        dgamma = np.sum(dout * self.x_norm, axis=0)
        dx_norm = dout * self.gamma

        dvar = np.sum(dx_norm * self.x_centered, axis=0) * -0.5 * (self.std_inv ** 3)
        dmean = np.sum(dx_norm * -self.std_inv, axis=0) + dvar * np.mean(-2. * self.x_centered, axis=0)
        dx = dx_norm * self.std_inv + dvar * 2 * self.x_centered / N + dmean / N

        return dx, dgamma, dbeta