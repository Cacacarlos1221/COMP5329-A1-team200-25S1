# Neural Network Implementation for Multi-class Classification

## Project Structure
This project implements a neural network for multi-class classification with various configurable components and experimental features.

### Core Components
- `neuralNetwork.py`: Core implementation of the neural network architecture
- `experiment.py`: Main script for running experiments with different model configurations
- `bestPerformance.py`: Script for training and evaluating the best performing model configuration
- `dataPre.py`: Data preprocessing 
- `visualization.py`: Visualization tools for model performance and data analysis

### Modules Directory
The `Modules/` directory contains essential neural network components:
- `activationFunction.py`: Implementation of various activation functions (ReLU, Sigmoid, etc.)
- `lossFunction.py`: Loss function implementations
- `optimizationMethods.py`: Optimization algorithms (Adam, SGD, etc.)
- `regularizationTechniques.py`: Regularization methods (Dropout, L1/L2)
- `trainingStrategies.py`: Training-related utilities and strategies

### Dataset
The `Assignment1-Dataset/` directory contains the training and testing datasets:
- `train_data.npy`: Training data
- `train_label.npy`: Training labels
- `test_data.npy`: Test data
- `test_label.npy`: Test labels

### Visualization Results
The `figure/` directory stores generated plots and visualizations:
- Model performance comparisons
- Data distribution analysis
- Training metrics visualization

## Running Instructions

### 1. Install Dependencies
```bash
pip install -r requirement.txt
```

### 2. Run Experiments
First, run the experiments to compare different model configurations:
```bash
python experiment.py
```
This will:
- Test various model configurations
- Generate performance comparison plots in the `figure/` directory
- Save the best configuration parameters

### 3. Train Best Model
After identifying the best configuration, run:
```bash
python bestPerformance.py
```
This will:
- Train the model using the best-performing configuration
- Evaluate the model on the test dataset
- Generate final performance metrics

## Results
The experimental results and visualizations will be saved in the `figure/` directory, allowing you to analyze:
- Performance comparisons across different configurations
- Training and validation metrics
- Data distribution and preprocessing effects

The best model configuration parameters will be saved in `saved_model_configurations.csv` for reference and reproducibility.