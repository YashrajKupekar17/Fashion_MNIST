# Fashion MNIST Classification with PyTorch, Optuna & MLflow

## Introduction
This project represents my first exploration of PyTorch, Optuna for hyperparameter tuning, and MLflow for experiment tracking. I built a neural network classifier for the Fashion MNIST dataset, achieving 90.1% accuracy through systematic hyperparameter optimization.

## About the Dataset
The Fashion MNIST dataset consists of 70,000 grayscale images (60,000 for training, 10,000 for testing) of 10 categories of clothing items. Each image is 28x28 pixels, making it a good starting point for computer vision tasks.

## Project Highlights
- **First PyTorch Implementation**: Built a custom neural network with dynamic architecture
- **First Use of Optuna**: Automated hyperparameter tuning to optimize model performance
- **First Integration with MLflow**: Tracked experiments and metrics for better reproducibility
- **Strong Results**: Achieved 90.1% accuracy on the test set

## Methodology
1. **Data Preparation**: Loaded and preprocessed the Fashion MNIST dataset
2. **Model Building**: Created a flexible neural network architecture in PyTorch
3. **Hyperparameter Optimization**: Used Optuna to find the optimal hyperparameters
4. **Experiment Tracking**: Utilized MLflow to track all experiments and metrics

## Optimal Parameters
After multiple trials with Optuna, the best model achieved 90.1% accuracy with:
- 5 hidden layers with 112 neurons per layer
- Learning rate: 0.00043
- Dropout rate: 0.1
- Optimizer: Adam
- Batch size: 64
- Weight decay: 5.25e-05
- Training epochs: 40

## Technical Challenges & Learnings
- Understanding PyTorch's autograd mechanism
- Implementing proper batching with DataLoader
- Setting up Optuna for efficient hyperparameter search
- Configuring MLflow for experiment tracking

## Running the Project
[Instructions on how to run your project...]

## Dependencies
- PyTorch
- Optuna
- MLflow
- pandas
- scikit-learn
- matplotlib

## Future Improvements
- Implement data augmentation techniques
- Explore more complex architectures like CNNs
- Deploy the model with a simple web interface
