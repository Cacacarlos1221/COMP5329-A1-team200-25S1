import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

class Visualizer:
    @staticmethod
    def plot_comparison_dual_metrics(results, title_prefix='', save_dir='figure'):
        """
        Plot training and validation performance for different experimental configurations.

        Args:
            results (dict): 
                Dictionary where keys are config names (e.g., 'relu') and values are dicts with keys 
                'accuracy', 'val_accuracy', 'loss', 'val_loss'.

            title_prefix (str): 
                Used in plot titles and filenames, e.g., "Activation Functions".

            save_dir (str): 
                Path to save output figures (assumed to exist).
        """

        # colour map for consistent colouring across plots
        color_map = plt.get_cmap("tab10")

        # === Accuracy Plot ===
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
        ax.xaxis.set_major_locator(MultipleLocator(1))  # Major ticks every 1 epoch
        ax.xaxis.set_minor_locator(plt.NullLocator())
        plt.xlim(1, len(epochs))
        plt.tight_layout(pad=0.5)
        plt.savefig(f'{save_dir}/{title_prefix.lower().replace(" ", "_")}_accuracy.png')
        plt.close()

        # === Loss Plot ===
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
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(plt.NullLocator())
        plt.xlim(1, len(epochs))
        plt.tight_layout(pad=0.5)
        plt.savefig(f'{save_dir}/{title_prefix.lower().replace(" ", "_")}_loss.png')
        plt.close()
