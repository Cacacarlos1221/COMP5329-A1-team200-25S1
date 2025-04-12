import numpy as np

class SGD:
    def __init__(self, learning_rate=0.01, momentum=0.9):
        """
        Initialize the Stochastic Gradient Descent optimizer with momentum.

        Args:
            learning_rate (float): Step size for parameter updates.
            momentum (float): Momentum factor (typically between 0 and 1).
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = {}  # for momentum, initialize to zero

    def update(self, params, grads):
        """
        Apply momentum-based SGD to update parameters.

        Args:
            params (dict): Dictionary of model parameters to be updated.
            grads (dict): Dictionary of gradients for each parameter.
        """
        for key in params.keys():
            # Initialize momentum if not already present
            if key not in self.velocities:
                self.velocities[key] = np.zeros_like(params[key])

            # Update momentum: v = momentum * v - learning_rate * grad
            self.velocities[key] = self.momentum * self.velocities[key] - self.learning_rate * grads[key]

            # Update parameter: param += v
            params[key] += self.velocities[key]
        return params

class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """
        Initialize the Adam optimizer.

        Args:
            learning_rate (float): Step size for parameter updates.
            beta1 (float): Exponential decay rate for the first moment estimate.
            beta2 (float): Exponential decay rate for the second moment estimate.
            epsilon (float): Small constant to prevent division by zero.
        """

        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, params, grads):
        """
        Apply Adam optimization to update parameters.

        Args:
            params (dict): Dictionary of model parameters to be updated.
            grads (dict): Dictionary of gradients for each parameter.
        """

        if not self.m:
            for key in params.keys():
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])

        self.t += 1

        for key in params.keys():

            # Update biased first moment estimate
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            
            # Update biased second raw moment estimate
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)

            # Compute bias-corrected moment estimates
            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            # Update parameters
            params[key] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return params