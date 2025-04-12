import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from neuralNetwork import NeuralNetwork
from dataPre import prepare_data
from visualization import Visualizer
from Modules.trainingStrategies import MiniBatchGenerator


class BestModelTrainer:
    def __init__(self):
        """
        Initialize the model trainer with training, validation, and test datasets.
        Also set up grid search hyperparameter configurations.
        """

        self.train_data, self.train_labels, self.val_data, self.val_labels, self.test_data, self.test_labels = prepare_data()
        self.num_epochs = 20

        # Define the search space for hyperparameter tuning
        self.param_grid = {
            'activation': ['leaky_relu', 'gelu'],
            'regularization': ['no_reg', 'batch_norm', 'batch_norm_dropout'],
            'batch_size': [32, 64],
            'hidden_layer': ['medium', 'large'],
            'learning_rate': [0.001, 0.0001]
        }


        # Save training histories, configurations, and model objects
        self.saved_models = {}
        self.saved_configs = {}
        self.saved_model_objects = {}

    def grid_search(self):
        """
        Perform exhaustive grid search over all hyperparameter combinations.
        Save models that achieve at least 55% validation accuracy.
        """

        total_combinations = (len(self.param_grid['activation']) *
                              len(self.param_grid['regularization']) *
                              len(self.param_grid['batch_size']) *
                              len(self.param_grid['hidden_layer']) *
                              len(self.param_grid['learning_rate']))
        current_combination = 0

        print(f"Starting grid search, total {total_combinations} combinations")

        for activation in self.param_grid['activation']:
            for regularization in self.param_grid['regularization']:
                for batch_size in self.param_grid['batch_size']:
                    for hidden_layer in self.param_grid['hidden_layer']:
                        for learning_rate in self.param_grid['learning_rate']:
                            current_combination += 1
                            print(f"\nTesting combination {current_combination}/{total_combinations}")
                            print(f"activation={activation}, regularization={regularization}, batch_size={batch_size}")
                            print(f"hidden_layer={hidden_layer}, learning_rate={learning_rate}\n")

                            # Define model architecture based on hidden layer size
                            layer_sizes = [128, 512, 256, 10] if hidden_layer == 'large' else [128, 256, 128, 10]
                            batch_generator = MiniBatchGenerator(batch_size=batch_size)

                            # Initialize neural network with current config
                            model = NeuralNetwork(
                                layer_sizes=layer_sizes,
                                activation=activation,
                                optimizer='adam',
                                dropout_rate=0.2 if regularization == 'batch_norm_dropout' else 0.0,
                                weight_decay_lambda=0.0,
                                use_batch_norm=(regularization in ['batch_norm', 'batch_norm_dropout']),
                                learning_rate=learning_rate
                            )

                            # Train and evaluate model
                            history = self.train_model(model, batch_generator)
                            val_loss, val_accuracy = model.evaluate(self.val_data, self.val_labels)

                            print(f"Validation accuracy: {val_accuracy:.4f}")

                            # Save models that meet accuracy threshold
                            if val_accuracy >= 0.55:
                                model_id = f'model_{len(self.saved_models) + 1}'
                                self.saved_models[model_id] = history
                                self.saved_configs[model_id] = {
                                    'activation': activation,
                                    'regularization': regularization,
                                    'batch_size': batch_size,
                                    'hidden_layer': hidden_layer,
                                    'learning_rate': learning_rate,
                                    'val_accuracy': val_accuracy
                                }
                                self.saved_model_objects[model_id] = model

        print("\nGrid search completed.")
        print(f"Total models saved (val_accuracy >= 0.55): {len(self.saved_models)}")

    def train_model(self, model, batch_generator):
        """
        Train a model using mini-batch gradient descent.

        Args:
            model: Neural network model to train.
            batch_generator: MiniBatchGenerator instance for data batching.
        """

        history = {'loss': [], 'accuracy': [], 'val_loss': [], 'val_accuracy': []}

        for epoch in range(self.num_epochs):
            epoch_loss = 0
            batch_count = 0

            # Loop over batches
            for x_batch, y_batch in batch_generator.generate_batches(self.train_data, self.train_labels):
                loss = model.train_step(x_batch, y_batch)
                epoch_loss += loss
                batch_count += 1

            # Track training and validation metrics
            train_loss, train_accuracy = model.evaluate(self.train_data, self.train_labels)
            val_loss, val_accuracy = model.evaluate(self.val_data, self.val_labels)

            history['loss'].append(train_loss)
            history['accuracy'].append(train_accuracy)
            history['val_loss'].append(val_loss)
            history['val_accuracy'].append(val_accuracy)

            print(f"Epoch {epoch + 1}/{self.num_epochs} - Train acc: {train_accuracy:.4f} - Val acc: {val_accuracy:.4f}")

        return history

    def show_and_plot_saved_models(self):
        """
        Plot and display all models with validation accuracy >= 0.55.
        Save their configurations and both accuracy/loss curves.
        """
        self.plot_comparison_dual_metrics(
            results=self.saved_models,
            title_prefix='High Accuracy Models',
            save_dir='figure'
        )

        df = pd.DataFrame.from_dict(self.saved_configs, orient='index')
        print("Saved Model Configurations:")
        print(df)
        df.to_csv("saved_model_configurations.csv", index=True)


    @staticmethod
    def plot_comparison_dual_metrics(results, title_prefix='', save_dir='figure'):
        """
        Plot both accuracy and loss comparison charts for different experimental configurations.

        Args:
            results (dict): Dictionary where each key is a model/experiment name (e.g., 'relu'),
                            and each value is a history dict with keys:
                            ['accuracy', 'val_accuracy', 'loss', 'val_loss'].
            title_prefix (str): Title prefix for the plots and filenames.
            save_dir (str): Directory where plots will be saved (assumed to already exist).
        """
        color_map = plt.get_cmap("tab10")

        # ---------------- Accuracy Plot ----------------
        plt.figure(figsize=(10, 5))
        for idx, (name, history) in enumerate(results.items()):
            color = color_map(idx % 10)
            epochs = range(1, len(history['accuracy']) + 1)
            plt.plot(epochs, history['accuracy'], label=f'{name} (train)', linestyle='-', color=color)
            plt.plot(epochs, history['val_accuracy'], label=f'{name} (val)', linestyle='--', color=color)

        plt.title(f'{title_prefix} Accuracy Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)

        ax = plt.gca()
        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.xaxis.set_minor_locator(plt.NullLocator())
        plt.xticks(range(1, len(epochs) + 1))
        plt.xlim(1, len(epochs))
        plt.tight_layout(pad=0.5)

        # Save accuracy plot
        acc_path = f'{save_dir}/{title_prefix.lower().replace(" ", "_")}_accuracy.png'
        plt.savefig(acc_path)
        plt.close()

        # ---------------- Loss Plot ----------------
        plt.figure(figsize=(10, 5))
        for idx, (name, history) in enumerate(results.items()):
            color = color_map(idx % 10)
            epochs = range(1, len(history['loss']) + 1)
            plt.plot(epochs, history['loss'], label=f'{name} (train)', linestyle='-', color=color)
            plt.plot(epochs, history['val_loss'], label=f'{name} (val)', linestyle='--', color=color)

        plt.title(f'{title_prefix} Loss Comparison')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        ax = plt.gca()
        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.xaxis.set_minor_locator(plt.NullLocator())
        plt.xticks(range(1, len(epochs) + 1))
        plt.xlim(1, len(epochs))
        plt.tight_layout(pad=0.5)

        # Save loss plot
        loss_path = f'{save_dir}/{title_prefix.lower().replace(" ", "_")}_loss.png'
        plt.savefig(loss_path)
        plt.close()


    def plot_single_model_performance(self, model_id, save_dir='figure'):

        """
        Plot training and validation accuracy/loss of a single selected model.

        Args:
            model_id (str): The model identifier to plot (e.g., 'model_2').
            save_dir (str): Directory to save the plots (assumes it exists).
        """

        history = self.saved_models[model_id]
        epochs = range(1, len(history['accuracy']) + 1)
        color = 'blue'

        # Accuracy plot
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, history['accuracy'], label='Train', linestyle='-', color=color)
        plt.plot(epochs, history['val_accuracy'], label='Validation', linestyle='--', color=color)
        plt.title(f'{model_id} Accuracy Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.xticks(epochs)
        plt.xlim(1, len(epochs))
        plt.tight_layout()
        plt.savefig(f'{save_dir}/best_model_accuracy.png')
        plt.close()

        # Loss plot
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, history['loss'], label='Train', linestyle='-', color=color)
        plt.plot(epochs, history['val_loss'], label='Validation', linestyle='--', color=color)
        plt.title(f'{model_id} Loss Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.xticks(epochs)
        plt.xlim(1, len(epochs))
        plt.tight_layout()
        plt.savefig(f'{save_dir}/best_model_loss.png')
        plt.close()

        print(f"Saved accuracy/loss plots for {model_id} to '{save_dir}'")



    def evaluate_selected_model(self, model_id='model_2'):
        """
        Evaluate a selected model on the test set.

        Args:
            model_id (str): Identifier of the saved model.
        """

        if model_id not in self.saved_model_objects:
            print(f"Model {model_id} not found.")
            return

        model = self.saved_model_objects[model_id]
        val_loss = self.saved_models[model_id]['val_loss'][-1]
        val_acc = self.saved_configs[model_id]['val_accuracy']
        test_loss, test_acc = model.evaluate(self.test_data, self.test_labels)

        print("\nSelected model:", model_id)
        print("Parameters:")
        for k, v in self.saved_configs[model_id].items():
            if k != 'val_accuracy':
                print(f"{k}: {v}")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_acc:.4f}")
        
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Accuracy: {test_acc:.4f}")
        self.plot_single_model_performance(model_id=model_id)


    def run_search_and_save(self):
        """
        Run grid search and save valid models and plots.
        """

        print("Starting grid search and saving models...")
        self.grid_search()
        self.show_and_plot_saved_models()

    def run_evaluation_for(self, model_id=None):
        """
        Prompt user or use given ID to evaluate a saved model.

        Args:
            model_id (str): Model identifier to evaluate. If None, prompt the user.
        """

        if model_id is None:
            model_id = input("Enter model ID to evaluate (e.g., model_1): ").strip()
        self.evaluate_selected_model(model_id=model_id)



if __name__ == '__main__':
    trainer = BestModelTrainer()
    trainer.run_search_and_save()
    trainer.run_evaluation_for()  
