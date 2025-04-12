import numpy as np

def ReLU(x, derivative=False):
    """
    Apply the ReLU activation function or its derivative.
    
    Args:
        x (ndarray): Input data.
        derivative (bool): Whether to compute the derivative. Default is False.
    """

    if derivative:
        return (x > 0).astype(float)
    return np.maximum(0, x)

def LeakyReLU(x, alpha=0.01, derivative=False):
    """
    Apply the LeakyReLU activation function or its derivative.
    
    Args:
        x (ndarray): Input data.
        alpha (float): Slope for negative input values.
        derivative (bool): Whether to compute the derivative. Default is False.
    """

    if derivative:
        return np.where(x > 0, 1, alpha)
    return np.where(x > 0, x, alpha * x)

def GELU(x, derivative=False):
    """
    Apply the approximate GELU activation function or its derivative.
    
    Args:
        x (ndarray): Input data.
        derivative (bool): Whether to compute the derivative. Default is False.
    """
    
    tanh_arg = np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)
    tanh_out = np.tanh(tanh_arg)
    if derivative:
        left = 0.5 * (1 + tanh_out)
        sech2 = 1 - tanh_out ** 2
        right = 0.5 * x * sech2 * np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * x**2)
        return left + right
    return 0.5 * x * (1 + tanh_out)