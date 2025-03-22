# Fashion MNIST Classification with PyTorch, Optuna & MLflow

## Project Report

**Author:** Yashraj Kupekar 
**Date:** March 22, 2025

## Executive Summary

This project implements a neural network classifier for the Fashion MNIST dataset using PyTorch. The model architecture and hyperparameters were optimized using Optuna, and all experiments were tracked using MLflow. The best model achieved 90.1% accuracy on the test set. This represents my first implementation using PyTorch, Optuna, and MLflow technologies.

## 1. Introduction

### 1.1 Problem Statement

The Fashion MNIST dataset has become a standard benchmark in computer vision, consisting of 70,000 grayscale images of clothing items across 10 categories. The goal of this project was to build an accurate classifier for these images while learning three important technologies in the machine learning ecosystem: PyTorch for neural network implementation, Optuna for hyperparameter optimization, and MLflow for experiment tracking.

### 1.2 Dataset Overview

Fashion MNIST contains 60,000 training images and 10,000 test images. Each image is a 28×28 grayscale image, associated with a label from 10 classes:

| Label | Description |
|-------|-------------|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

## 2. Methodology

### 2.1 Data Preprocessing

The dataset was loaded and preprocessed with the following steps:
- Normalization of pixel values from [0, 255] to [0, 1]
- Train-test split (80% training, 20% testing)
- Creation of custom PyTorch datasets and data loaders for batch processing

### 2.2 Model Architecture

A customizable neural network architecture was implemented with the following components:
- Flexible number of hidden layers
- Configurable number of neurons per layer
- Batch normalization after each linear layer
- ReLU activation functions
- Dropout for regularization
- Final layer with 10 output neurons (one per class)

The model accepts 784 input features (28×28 pixels flattened) and produces probabilities for each of the 10 clothing categories.

### 2.3 Hyperparameter Optimization

Optuna was used to systematically search for optimal hyperparameters. The following parameters were tuned:

| Hyperparameter | Search Range | Description |
|----------------|--------------|-------------|
| Number of layers | 2-5 | Number of hidden layers in the network |
| Neurons per layer | 8-128 | Number of neurons in each hidden layer |
| Learning rate | 1e-5 to 1e-1 | Step size for optimizer updates |
| Epochs | 10-50 | Number of training cycles |
| Dropout rate | 0.1-0.5 | Probability of neuron deactivation |
| Optimizer | Adam, SGD, RMSprop | Algorithm to update weights |
| Batch size | 16, 32, 64, 128 | Number of samples per gradient update |
| Weight decay | 1e-5 to 1e-3 | L2 regularization strength |

### 2.4 Experiment Tracking

MLflow was integrated to track all experimental runs. For each trial, the following were logged:
- All hyperparameters
- Model accuracy
- Training time

## 3. Results

### 3.1 Optimal Configuration

After 25 trials, Optuna found the following optimal configuration:

| Parameter | Optimal Value |
|-----------|---------------|
| Number of layers | 5 |
| Neurons per layer | 112 |
| Learning rate | 0.00043 |
| Epochs | 40 |
| Dropout rate | 0.1 |
| Optimizer | Adam |
| Batch size | 64 |
| Weight decay | 5.25e-05 |

### 3.2 Model Performance

The best model achieved **90.1%** accuracy on the test set. This result is competitive for a fully connected neural network implementation without using convolutional layers.

## 4. Discussion

### 4.1 Key Findings

1. **Architecture Impact**: The number of layers and neurons significantly impacted model performance. The optimal configuration used a deeper network (5 layers) with a relatively large number of neurons (112 per layer).

2. **Regularization**: A relatively low dropout rate (0.1) combined with a small weight decay (5.25e-05) provided the best balance for regularization.

3. **Optimizer Choice**: Adam consistently outperformed SGD and RMSprop in trials, likely due to its adaptive learning rate mechanics.

4. **Batch Size Influence**: Moderate batch sizes (64) provided the best balance between computational efficiency and gradient quality.

### 4.2 Challenges Encountered

1. **PyTorch Learning Curve**: Understanding PyTorch's tensor operations and computational graph mechanics required significant learning.

2. **Hyperparameter Interdependence**: Many hyperparameters interact with each other, making the optimization process complex.

3. **Computational Resources**: Running multiple Optuna trials was computationally intensive, requiring efficient resource management.

## 5. Technical Implementation

### 5.1 Custom Dataset Class

A custom PyTorch Dataset class was implemented to handle the Fashion MNIST data:

```python
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        super().__init__()
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, index):
        return self.features[index], self.labels[index]
```

### 5.2 Dynamic Neural Network Implementation

The neural network architecture was made flexible to accommodate Optuna's hyperparameter search:

```python
class MyNN(nn.Module):
    def __init__(self, input_dim, output_dim, num_of_layers, num_neurons_per_layer, dropout_rate):
        super().__init__()
        layers = []
        for i in range(num_of_layers):
            layers.append(nn.Linear(input_dim, num_neurons_per_layer))
            layers.append(nn.BatchNorm1d(num_neurons_per_layer))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))
            input_dim = num_neurons_per_layer
        layers.append(nn.Linear(num_neurons_per_layer, output_dim))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)
```

### 5.3 Optuna Trial Definition

The objective function for Optuna was implemented to construct, train, and evaluate models:

```python
def objective(trial):
    with mlflow.start_run(nested=True):
        # Hyperparameter selection
        num_of_layers = trial.suggest_int("num_of_layers", 2, 5)
        num_neurons_per_layer = trial.suggest_int("num_neurons_per_layer", 8, 128, step=8)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
        epochs = trial.suggest_int("epochs", 10, 50, step=10)
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.1)
        optimizer_name = trial.suggest_categorical("optimizer", ['Adam', 'SGD', 'RMSprop'])
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
        
        # Log parameters with MLflow
        mlflow.log_param("num_layers", num_of_layers)
        mlflow.log_param("num_neurons_per_layer", num_neurons_per_layer)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("dropout_rate", dropout_rate)
        mlflow.log_param("optimizer_name", optimizer_name)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("weight_decay", weight_decay)
        
        # Model training and evaluation
        # [... training code omitted for brevity ...]
        
        # Log accuracy metric
        mlflow.log_metric("accuracy", accuracy)
        return accuracy
```

## 6. Conclusion and Future Work

### 6.1 Conclusion

This project successfully demonstrated the application of PyTorch for implementing neural networks, Optuna for hyperparameter optimization, and MLflow for experiment tracking. Despite being a first-time implementation with these technologies, a competitive accuracy of 90.1% was achieved on the Fashion MNIST dataset.

The combination of these technologies provided a systematic approach to model development and optimization, creating a robust workflow that could be applied to other machine learning problems.

### 6.2 Future Work

Several directions for future improvement and exploration include:

1. **Implementing Convolutional Neural Networks (CNNs)**: CNNs are better suited for image classification tasks and could significantly improve performance.

2. **Data Augmentation**: Introducing techniques like rotation, flipping, and zooming to increase the effective size of the training dataset.

3. **Ensemble Methods**: Combining multiple models to improve prediction accuracy.

4. **Model Deployment**: Creating a simple web application to demonstrate the model in a production-like environment.

5. **Transfer Learning**: Exploring the use of pre-trained models for feature extraction.

## 7. References

1. Fashion MNIST Dataset: https://github.com/zalandoresearch/fashion-mnist
2. PyTorch Documentation: https://pytorch.org/docs/stable/index.html
3. Optuna Documentation: https://optuna.readthedocs.io/
4. MLflow Documentation: https://mlflow.org/docs/latest/index.html

---

## Experiment Tracking Visualization

Link to the Dagshub experiment tracking dashboard: https://dagshub.com/yashraj4ml/Fashion_mnist/experiments


