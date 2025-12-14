# Linear Regression from Scratch

## Overview
This project implements linear regression from scratch using gradient descent optimization, without relying on scikit-learn or other machine learning libraries. It demonstrates the fundamental concepts of supervised learning and optimization.

## Problem Description
Linear regression models the relationship between input features (X) and output values (y) using a linear equation:

**y = w·X + b**

Where:
- **w**: weight (slope) parameter
- **b**: bias (intercept) parameter
- **X**: input features
- **y**: target/output values

## Dataset
A synthetic dataset is generated for this project:
- **Samples**: 100 data points
- **True Relationship**: y = 2.5·X + 5.0 + noise
- **X Range**: Random values between 0 and 10
- **Noise**: Gaussian noise to simulate real-world data

## Algorithm Components

### 1. Model Function
```
predict(X, w, b) = X·w + b
```
Computes predictions based on current parameters.

### 2. Loss Function
**Mean Squared Error (MSE)**:
```
Loss = (1/n) × Σ(y_true - y_pred)²
```
Measures the average squared difference between predictions and actual values.

### 3. Gradient Descent Optimization
Updates parameters to minimize the loss function:

**Weight Gradient**: dw = -2·Σ(X·(y - y_pred))/n

**Bias Gradient**: db = -2·Σ(y - y_pred)/n

**Parameter Updates**:
- w = w - α·dw
- b = b - α·db

Where α is the learning rate.

## Implementation Details

### Hyperparameters
- **Learning Rate (α)**: 0.01
- **Epochs**: 100 training iterations
- **Random Seed**: 42 (for reproducibility)

### Training Process
1. Initialize weights and bias randomly
2. For each epoch:
   - Compute predictions using current parameters
   - Calculate loss (MSE)
   - Compute gradients of loss w.r.t. parameters
   - Update parameters using gradients
   - Log loss every 10 epochs

### Visualization
The notebook includes three key visualizations:
1. **Original Dataset**: Scatter plot showing the synthetic data distribution
2. **Training Loss**: Line plot showing loss decrease over epochs
3. **Model Predictions**: Comparison of training data with learned regression line

## Key Concepts Demonstrated

### 1. Supervised Learning
- Learning from labeled data (X, y pairs)
- Model learns to map inputs to outputs

### 2. Gradient Descent
- Iterative optimization algorithm
- Follows the negative gradient to minimize loss
- Learning rate controls step size

### 3. Convergence
- Loss decreases over epochs
- Model parameters converge to optimal values
- Final parameters approximate true relationship

### 4. Mean Squared Error
- Differentiable loss function
- Penalizes large errors more than small errors
- Convex for linear regression (guaranteed convergence)

## Mathematical Foundation

### Cost Function Derivation
The partial derivatives used in gradient descent:

**∂Loss/∂w** = (2/n) × Σ(-X·(y - y_pred))

**∂Loss/∂b** = (2/n) × Σ(-(y - y_pred))

These gradients point in the direction of steepest ascent, so we subtract them (gradient descent) to minimize loss.

## Results
After training:
- The model learns parameters close to the true values (w≈2.5, b≈5.0)
- The regression line fits the data well
- Loss decreases smoothly during training
- Test predictions align with expected outputs

## Requirements
- NumPy (for numerical computations)
- Matplotlib (for visualization)

## Usage
Run cells sequentially to:
1. Generate synthetic dataset
2. Visualize data distribution
3. Initialize model parameters
4. Define prediction and loss functions
5. Implement gradient descent
6. Train the model
7. Visualize training progress and results

## Extensions
This basic implementation can be extended to:
- Multiple features (multivariate linear regression)
- Polynomial features (polynomial regression)
- Regularization (Ridge/Lasso regression)
- Mini-batch or stochastic gradient descent
- Adaptive learning rates (momentum, Adam)

## Learning Outcomes
- Understanding of linear regression mathematics
- Implementation of gradient descent from scratch
- Loss function optimization
- Parameter initialization and convergence
- Data visualization and model evaluation
- Foundation for understanding more complex models
