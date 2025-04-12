import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

class DataPreprocessor:
    def __init__(self):
        """
        Initialize the data preprocessor (no stats needed for raw input).
        """
        self.mean = None
        self.std = None
    
    def load_data(self, train_data_path, train_label_path, test_data_path=None, test_label_path=None, val_split=0.2):
        """
        Load training and test datasets, shuffle, and split validation set.

        Args:
            train_data_path: Path to training data (.npy)
            train_label_path: Path to training labels (.npy)
            test_data_path: Optional path to test data (.npy)
            test_label_path: Optional path to test labels (.npy)
            val_split: Proportion of training data for validation
        """
        train_data = np.load(train_data_path)
        train_labels = np.load(train_label_path)
        

        indices = np.arange(train_data.shape[0])
        np.random.shuffle(indices)
        train_data = train_data[indices]
        train_labels = train_labels[indices]
        

        val_size = int(train_data.shape[0] * val_split)
        val_data = train_data[:val_size]
        val_labels = train_labels[:val_size]
        train_data = train_data[val_size:]
        train_labels = train_labels[val_size:]
        
        test_data = None
        test_labels = None
        if test_data_path and test_label_path:
            test_data = np.load(test_data_path)
            test_labels = np.load(test_label_path)
        
        return train_data, train_labels, val_data, val_labels, test_data, test_labels
    
    
    def visualize_data(self, data, labels, num_samples=5, save_path=None):
        """
        Visualize a few input samples.

        Args:
            data: Input features.
            labels: Class labels.
            num_samples: Number of samples to visualize.
            save_path: Optional path to save the figure.
        """

        plt.figure(figsize=(15, 3))
        for i in range(num_samples):
            plt.subplot(1, num_samples, i + 1)
            if data[i].size == 128:
                img_data = data[i].reshape(8, 16)
            else:
                side = int(np.sqrt(data[i].size))
                img_data = data[i].reshape(side, -1)
            plt.imshow(img_data, cmap='gray')
            plt.title(f'Label: {labels[i]}')
            plt.axis('off')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def visualize_data_distribution(self, labels, save_path=None):
        """
        Plot the distribution of labels (class frequencies).

        Args:
            labels: Array of class labels.
            save_path: Optional path to save the figure.
        """

        unique_labels, counts = np.unique(labels, return_counts=True)
        
        plt.figure(figsize=(10, 6))
        plt.bar(unique_labels, counts)
        plt.title('Data Distribution')
        plt.xlabel('Label')
        plt.ylabel('Sample number')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def standardize(self, train_data, test_data=None):
        """
        standardize data
        """
        if self.mean is None:
            self.mean = np.mean(train_data, axis=0)
            self.std = np.std(train_data, axis=0)
        
        train_data_std = (train_data - self.mean) / (self.std + 1e-8)
        
        if test_data is not None:
            test_data_std = (test_data - self.mean) / (self.std + 1e-8)
            return train_data_std, test_data_std
        
        return train_data_std
    
    def normalize(self, train_data, test_data=None):
        """
        normalize data
        """
        if self.min_val is None:
            self.min_val = np.min(train_data, axis=0)
            self.max_val = np.max(train_data, axis=0)
        
        train_data_norm = (train_data - self.min_val) / (self.max_val - self.min_val + 1e-8)
        
        if test_data is not None:
            test_data_norm = (test_data - self.min_val) / (self.max_val - self.min_val + 1e-8)
            return train_data_norm, test_data_norm
        
        return train_data_norm
        
    def dist_graph(self, data, save_path=None):
        """
        Plot histogram and fitted normal distribution curve.

        Args:
            data: 1D array of values to plot.
            save_path: Optional path to save the plot.
        """
        
        plt.figure(figsize=(10, 6))
        

        counts, bins, patches = plt.hist(data, bins=30, density=True, 
                                        alpha=0.6, color='g', edgecolor='black')
        
        mu = np.mean(data)
        sigma = np.std(data)
        
        x = np.linspace(bins[0], bins[-1], 100)
        pdf = stats.norm.pdf(x, mu, sigma)
        
        plt.plot(x, pdf, 'k-', linewidth=2, label='Fitted Normal PDF')
        
        plt.title('Data Distribution')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def visualize_qqplot(data, dist='norm', title="Normal Q-Q", save_path=None):
        """
        Create a Q-Q plot to assess distributional fit.

        Args:
            data: 1D array of data to test.
            dist: Theoretical distribution (default is normal).
            title: Plot title.
            save_path: Optional path to save the figure.
        """

        plt.figure(figsize=(10, 6))
        stats.probplot(data, dist=dist, plot=plt)
        plt.title(title or 'QQ Plot')
        plt.xlabel('theoretical quantiles')
        plt.ylabel('sample quantiles')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

def prepare_data(val_split=0.2):
    """
    Load and prepare data for training, including visualization and preprocessing.

    Args:
        val_split: Proportion of training data to use as validation set.
    """
    preprocessor = DataPreprocessor()
    

    train_data, train_labels, val_data, val_labels, test_data, test_labels = preprocessor.load_data(
        'Assignment1-Dataset/train_data.npy',
        'Assignment1-Dataset/train_label.npy',
        'Assignment1-Dataset/test_data.npy',
        'Assignment1-Dataset/test_label.npy',
        val_split=val_split
    )

    
    # data normalization
    train_data_nor = preprocessor.standardize(train_data)
    val_data_nor = preprocessor.standardize(val_data)
    test_data_nor = preprocessor.standardize(test_data) if test_data is not None else None
    # data standardization
    train_data_std = preprocessor.standardize(train_data)
    val_data_std = preprocessor.standardize(val_data)
    test_data_std = preprocessor.standardize(test_data) if test_data is not None else None


    # visualization samples
    preprocessor.visualize_data(train_data, train_labels, save_path='figure/sample_images.png')
    preprocessor.visualize_data(train_data_nor, train_labels, save_path='figure/sample_images_nor.png')
    preprocessor.visualize_data(train_data_std, train_labels, save_path='figure/sample_images_std.png')
    
    
    preprocessor.visualize_data_distribution(train_labels, save_path='figure/data_distribution.png')
    preprocessor.visualize_qqplot(train_data.flatten(), save_path='figure/normalQQ_plot.png')
    preprocessor.dist_graph(train_data.flatten(), save_path='figure/dist_graph.png')
    
    return train_data, train_labels, val_data, val_labels, test_data, test_labels 

if __name__ == '__main__':
    prepare_data()

