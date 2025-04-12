import numpy as np
from neuralNetwork import NeuralNetwork
from dataPre import prepare_data
from visualization import Visualizer
from Modules.trainingStrategies import MiniBatchGenerator

class ExperimentRunner:
    def __init__(self):

        """
        Initialize experiment runner by loading dataset, setting up visualizer and batch generator.
        """

        self.train_data, self.train_labels, self.val_data, self.val_labels, self.test_data, self.test_labels = prepare_data()
        self.visualizer = Visualizer()
        self.batch_generator = MiniBatchGenerator(batch_size=32)
        self.num_epochs = 20
    
    def train_model(self, model, experiment_name):
        """
        Train the given model for a fixed number of epochs and record training/validation metrics.

        Args:
            model: NeuralNetwork instance.
            experiment_name: Identifier for logging during training.
        """

        history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}
        
        for epoch in range(self.num_epochs):
            epoch_loss = 0
            batch_count = 0
            
            # train
            for x_batch, y_batch in self.batch_generator.generate_batches(self.train_data, self.train_labels):
                loss = model.train_step(x_batch, y_batch)
                epoch_loss += loss
                batch_count += 1
            
            # train loss and train accuracy
            train_loss, train_accuracy = model.evaluate(self.train_data, self.train_labels)
            
            # validation loss and validation accuracy
            val_loss, val_accuracy = model.evaluate(self.val_data, self.val_labels)
            
            # record history
            history['loss'].append(train_loss)
            history['accuracy'].append(train_accuracy)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)
            
            # calcualte average epoch loss
            avg_epoch_loss = epoch_loss / batch_count if batch_count > 0 else 0
            
            # print information
            print(f'\n{experiment_name} - Epoch {epoch+1}/{self.num_epochs}', flush=True)
            print(f'Average batch loss: {avg_epoch_loss:.4f}', flush=True)
            print(f'Training - loss: {train_loss:.4f} - accuracy: {train_accuracy:.4f}', flush=True)
            print(f'Validation - loss: {val_loss:.4f} - accuracy: {val_accuracy:.4f}', flush=True)
            print('-' * 50, flush=True)
        
        return history
    
    def compare_activations(self):
        """
        compare different activation functions
        """
        activations = ['relu', 'leaky_relu', 'gelu']
        results = {}
        
        for activation in activations:
            model = NeuralNetwork(
                layer_sizes=[128, 256, 128, 10],
                activation=activation
            )
            history = self.train_model(model, f'Activation: {activation}')
            results[activation] = history

        self.visualizer.plot_comparison_dual_metrics(
            results,
            title_prefix='Activation Functions'
        )

        return results
    
    def compare_optimizers(self):
        """
        compare different optimizers
        """
        optimizers = ['sgd', 'adam']
        results = {}
        
        for optimizer in optimizers:
            model = NeuralNetwork(
                layer_sizes=[128, 256, 128, 10],
                optimizer=optimizer
            )
            history = self.train_model(model, f'Optimizer: {optimizer}')
            results[optimizer] = history
        

        self.visualizer.plot_comparison_dual_metrics(
            results,
            title_prefix='Optimizers'
        )


        return results
    
    def compare_regularization(self):
        """
        compare different regularization techniques
        """
        configs = {
            'no_reg': {'dropout_rate': 0.0, 'weight_decay_lambda': 0.0, 'use_batch_norm': False},
            'dropout': {'dropout_rate': 0.2, 'weight_decay_lambda': 0.0, 'use_batch_norm': False},
            'weight_decay': {'dropout_rate': 0.0, 'weight_decay_lambda': 0.01, 'use_batch_norm': False},
            'batch_norm': {'dropout_rate': 0.0, 'weight_decay_lambda': 0.0, 'use_batch_norm': True},
            'batch_norm_dropout': {'dropout_rate': 0.2, 'weight_decay_lambda': 0.0, 'use_batch_norm': True},
            'all': {'dropout_rate': 0.2, 'weight_decay_lambda': 0.01, 'use_batch_norm': True}
        }
        
        results = {}
        for name, config in configs.items():
            model = NeuralNetwork(
                layer_sizes=[128, 256, 128, 10],
                **config
            )
            history = self.train_model(model, f'Regularization: {name}')
            results[name] = history
        

        self.visualizer.plot_comparison_dual_metrics(
            results,
            title_prefix='Regularization Techniques'
        )

        
        return results

    def compare_dropout_rates(self):
        """
        compares the performance of different dropout rates
        """
        dropout_rates = [0.0, 0.1, 0.3, 0.5]
        results = {}

        for rate in dropout_rates:
            model = NeuralNetwork(
                layer_sizes=[128, 256, 128, 10],
                dropout_rate=rate,
                weight_decay_lambda=0.0  
            )
            history = self.train_model(model, f'Dropout Rate: {rate}')
            results[f'dropout_{rate}'] = history


        self.visualizer.plot_comparison_dual_metrics(
            results,
            title_prefix='Dropout Rates'
        )

        return results


    def compare_weight_decay_strength(self):
        """
        Compare different L2 regularization strengths.
        """

        lambdas = [0.0, 0.001, 0.01, 0.1]
        results = {}

        for lam in lambdas:
            model = NeuralNetwork(
                layer_sizes=[128, 256, 128, 10],
                weight_decay_lambda=lam,
                dropout_rate=0.0
            )
            history = self.train_model(model, f'Weight Decay λ={lam}')
            results[f'lambda_{lam}'] = history


        self.visualizer.plot_comparison_dual_metrics(
            results,
            title_prefix='Weight Decay Strength'
        )

        return results
    
    
    def compare_batch_sizes(self):
        """
        compare different batch sizes with the same other hyperparameters
        """
        batch_sizes = [16, 32, 64, 128]
        results = {}
        
        for batch_size in batch_sizes:
            self.batch_generator = MiniBatchGenerator(batch_size=batch_size)
            model = NeuralNetwork(layer_sizes=[128, 256, 128, 10])
            history = self.train_model(model, f'Batch Size: {batch_size}')
            results[f'batch_{batch_size}'] = history


        self.visualizer.plot_comparison_dual_metrics(
            results,
            title_prefix='Batch Sizes'
        )

        return results
    
    def compare_learning_rates(self):
        """
        Compare different learning rates.
        """

        lrs = [0.0001, 0.001, 0.01, 0.1]
        results = {}

        for lr in lrs:
            model = NeuralNetwork(
                layer_sizes=[128, 256, 128, 10],
                learning_rate=lr
            )
            history = self.train_model(model, f'Learning Rate: {lr}')
            results[f'lr_{lr}'] = history


        self.visualizer.plot_comparison_dual_metrics(
            results,
            title_prefix='Learning Rates'
        )

        return results


    def compare_hidden_layer_sizes(self):
        """
        Compare different hidden layer configurations.
        """

        configs = {
            'small': [128, 64, 10],
            'medium': [128, 256, 128, 10],
            'large': [128, 512, 256, 128, 10]
        }
        results = {}

        for name, structure in configs.items():
            model = NeuralNetwork(
                layer_sizes=structure
            )
            history = self.train_model(model, f'Hidden Layer Config: {name}')
            results[name] = history


        self.visualizer.plot_comparison_dual_metrics(
            results,
            title_prefix='Hidden Layer Sizes'
        )

        return results




def run_experiments():
    """
    run experiments and save results
    """
    runner = ExperimentRunner()
    
    # record results
    results = {
        'activation': runner.compare_activations(),
        'optimizer': runner.compare_optimizers(),
        'regularization': runner.compare_regularization(),
        'batch_size': runner.compare_batch_sizes(),
        'dropout_rate': runner.compare_dropout_rates(),
        'weight_decay': runner.compare_weight_decay_strength(),
        'learning_rate': runner.compare_learning_rates(),
        'hidden_size': runner.compare_hidden_layer_sizes()
    }
    
    # print best results
    print('\nBest Validation Accuracies:')
    for exp_name, exp_results in results.items():
        best_config = max(exp_results.items(), key=lambda x: max(x[1]['val_accuracy']))
        print(f'{exp_name}: {best_config[0]} - {max(best_config[1]["val_accuracy"]):.4f}')
    
    return results

if __name__ == '__main__':
    run_experiments()