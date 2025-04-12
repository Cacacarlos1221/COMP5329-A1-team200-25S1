import numpy as np
from Modules.activationFunction import ReLU, LeakyReLU, GELU
from Modules.optimizationMethods import SGD, Adam
from Modules.regularizationTechniques import WeightDecay, Dropout
from Modules.lossFunction import SoftmaxCrossEntropy
from Modules.trainingStrategies import MiniBatchGenerator, BatchNormalization

class NeuralNetwork:
    def __init__(self, layer_sizes, activation='relu', optimizer='adam', 
                 dropout_rate=0, weight_decay_lambda=0,
                 learning_rate=0.001, use_batch_norm=False):
        """
        Initialize a configurable feedforward neural network.

        Args:
            layer_sizes (list): Sizes of each layer, including input and output. E.g., [128, 256, 10]
            activation (str): Type of activation ('relu', 'leaky_relu', 'gelu')
            optimizer (str): Optimization algorithm ('sgd' or 'adam')
            dropout_rate (float): Dropout rate to apply between layers
            weight_decay_lambda (float): L2 regularization coefficient
            learning_rate (float): Learning rate for optimizer
            use_batch_norm (bool): Whether to apply batch normalization
        """

        self.layer_sizes = layer_sizes
        self.params = {}
        self.batch_norms = {}
        self.dropouts = {}

        # Initialize weights and biases
        for i in range(len(layer_sizes)-1):
            self.params[f'W{i+1}'] = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0/layer_sizes[i])
            self.params[f'b{i+1}'] = np.zeros(layer_sizes[i+1])
            # Create batch normalization layer for each layer (if enabled)
            if use_batch_norm:
                self.batch_norms[f'bn{i+1}'] = BatchNormalization(layer_sizes[i+1])
            # Create Dropout for each layer
            self.dropouts[f'dropout{i+1}'] = Dropout(dropout_rate)

        # Set activation function
        self.activation_funcs = {
            'relu': ReLU,
            'leaky_relu': LeakyReLU,
            'gelu': GELU
        }
        self.activation = self.activation_funcs[activation]

        # Set optimizer
        if optimizer == 'sgd':
            self.optimizer = SGD(learning_rate)
        else:
            self.optimizer = Adam(learning_rate)

        # Set regularization
        self.weight_decay = WeightDecay(weight_decay_lambda)

        # Set loss function
        self.loss_function = SoftmaxCrossEntropy()

        # Cache intermediate results of forward propagation
        self.cache = {}

    def forward(self, x, training=True):
        """
        Forward pass through the network.

        Args:
            x (ndarray): Input data, shape (batch_size, input_dim)
            training (bool): Whether in training mode (affects dropout/batchnorm)
        """

        self.cache['a0'] = x

        for i in range(len(self.layer_sizes)-1):
            # Linear transformation
            z = np.dot(self.cache[f'a{i}'], self.params[f'W{i+1}']) + self.params[f'b{i+1}']

            # Batch normalization (if enabled)
            if hasattr(self, 'batch_norms') and f'bn{i+1}' in self.batch_norms:
                z = self.batch_norms[f'bn{i+1}'].forward(z, training)

            # Activation function
            a = self.activation(z)

            # Dropout
            if training:
                a = self.dropouts[f'dropout{i+1}'].forward(a, training)

            self.cache[f'z{i+1}'] = z
            self.cache[f'a{i+1}'] = a

        return self.cache[f'a{len(self.layer_sizes)-1}']

    def backward(self, y_true):
        """
        Backward pass to compute gradients.

        Args:
            y_true (ndarray): True labels, shape (batch_size,)
        """

        batch_size = y_true.shape[0]
        grads = {}

        # Compute output layer gradients (call backward method of loss_function)
        dout = self.loss_function.backward()

        # Traverse layers backwards
        for i in range(len(self.layer_sizes)-1, 0, -1):
            # Dropout backward propagation (except output layer)
            if i < len(self.layer_sizes)-1:
                dout = self.dropouts[f'dropout{i}'].backward(dout)

            # Activation function backward propagation
            dz = dout * self.activation(self.cache[f'z{i}'], derivative=True)

            # Batch normalization backward propagation (if enabled)
            if hasattr(self, 'batch_norms') and f'bn{i}' in self.batch_norms:
                dz, dgamma, dbeta = self.batch_norms[f'bn{i}'].backward(dz)

            # Compute gradients for weights and biases (linear layer gradients)
            grads[f'W{i}'] = np.dot(self.cache[f'a{i-1}'].T, dz) / batch_size
            grads[f'b{i}'] = np.mean(dz, axis=0)

            # Add L2 regularization gradients only to current layer weights
            reg_grad = self.weight_decay.compute_gradients({f'W{i}': self.params[f'W{i}']})
            grads[f'W{i}'] += reg_grad[f'W{i}']

            # Compute gradients passed to previous layer
            if i > 1:
                dout = np.dot(dz, self.params[f'W{i}'].T)

        # Reset internal state of loss_function to avoid state confusion
        self.loss_function.y_pred = None
        self.loss_function.y_true_one_hot = None

        return grads



    def train_step(self, x_batch, y_batch):
        """
        Run one training step: forward, loss, backward, update.

        Args:
            x_batch (ndarray): Mini-batch input
            y_batch (ndarray): Mini-batch labels
        """
        # Forward propagation
        y_pred = self.forward(x_batch, training=True)

        # Compute loss
        loss = self.loss_function.forward(y_pred, y_batch)
        reg_loss = self.weight_decay.compute_penalty(self.params)
        total_loss = loss + reg_loss

        # Backward propagation
        grads = self.backward(y_batch)

        # Update parameters
        self.params = self.optimizer.update(self.params, grads)

        return total_loss

    def predict(self, x):
        """
        Predict labels from input (in inference mode).

        Args:
            x (ndarray): Input data
        """
        return self.forward(x, training=False)

    def evaluate(self, x, y):
        """
        Evaluate model performance.

        Args:
            x (ndarray): Input data
            y (ndarray): True labels
        """
        y_pred = self.predict(x)
        loss = self.loss_function.forward(y_pred, y)
        accuracy = self.loss_function.accuracy(y_pred, y)
        return loss, accuracy