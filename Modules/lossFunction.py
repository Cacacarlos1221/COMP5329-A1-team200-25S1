import numpy as np

class SoftmaxCrossEntropy:
    def __init__(self):
        """
        Initialize storage for predictions and true labels.
        """

        self.y_pred = None
        self.y_true = None

    def softmax(self, logits):
        """
        Compute the softmax of the input logits in a numerically stable way.

        Args:
            logits: Raw model outputs, shape (batch_size, num_classes).
        """

        logits = logits - np.max(logits, axis=1, keepdims=True)  
        exp_logits = np.exp(logits)
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

    def forward(self, logits, y_true):
        """
        Compute softmax probabilities and cross-entropy loss.

        Args:
            logits: Raw model outputs, shape (batch_size, num_classes).
            y_true: Ground truth labels, shape (batch_size,), with integer class indices.
        """

        self.y_true = y_true.reshape(-1)  
        self.y_pred = self.softmax(logits)

        batch_indices = np.arange(len(self.y_true))
        correct_class_probs = self.y_pred[batch_indices, self.y_true]

        loss = -np.mean(np.log(np.clip(correct_class_probs, 1e-15, 1.0)))
        return loss

    def backward(self):
        """
        Compute the gradient of the loss with respect to the input logits.
        """

        grad = self.y_pred.copy()
        batch_indices = np.arange(len(self.y_true))
        grad[batch_indices, self.y_true] -= 1
        grad /= len(self.y_true)
        return grad

    def accuracy(self, logits, y_true):
        """
        Compute classification accuracy.

        Args:
            logits: Raw model outputs, shape (batch_size, num_classes).
            y_true: Ground truth labels, shape (batch_size,).
        """
        
        preds = np.argmax(logits, axis=1)
        y_true = y_true.reshape(-1)
        return np.mean(preds == y_true)
