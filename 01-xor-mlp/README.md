# XOR Problem using Multi-Layer Perceptron (MLP)

## Overview
This project implements a neural network solution to the classic XOR (Exclusive OR) problem, which is a fundamental example in machine learning that demonstrates the importance of non-linear activation functions and hidden layers.

## Problem Description
The XOR problem is a binary classification problem where the output is 1 if and only if one (but not both) of the inputs is 1:
- Input: [0, 0] → Output: 0
- Input: [0, 1] → Output: 1
- Input: [1, 0] → Output: 1
- Input: [1, 1] → Output: 0

This problem cannot be solved by a single-layer perceptron (linear classifier) because the data is not linearly separable.

## Architecture
The implementation uses a simple Multi-Layer Perceptron with:
- **Input Layer**: 2 neurons (for 2 input features)
- **Hidden Layer**: 2 neurons with sigmoid activation
- **Output Layer**: 1 neuron with sigmoid activation

## Key Components

### Activation Function
- **Sigmoid Function**: σ(x) = 1 / (1 + e^(-x))
  - Outputs values between 0 and 1
  - Provides non-linearity needed to solve XOR
  - Derivative: σ'(x) = σ(x) × (1 - σ(x))

### Training Process
1. **Forward Propagation**:
   - Compute hidden layer: h = σ(X·W1 + b1)
   - Compute output: y = σ(h·W2 + b2)

2. **Loss Calculation**: Mean Squared Error (MSE)
   - Loss = mean((y_true - y_pred)²)

3. **Backpropagation**:
   - Calculate gradients for output layer
   - Calculate gradients for hidden layer (using chain rule)
   - Update weights and biases using gradient descent

### Hyperparameters
- **Epochs**: 10,000 training iterations
- **Learning Rate**: 0.1
- **Weight Initialization**: Random uniform distribution between -1 and 1

## Implementation Details

### Key Functions
1. `sigmoid(x)`: Applies sigmoid activation
2. `sigmoid_derivative(x)`: Computes derivative for backpropagation

### Training Loop
- Performs forward pass through network
- Computes error between predictions and actual values
- Uses backpropagation to calculate gradients
- Updates weights and biases using gradient descent
- Prints loss every 1000 epochs for monitoring

## Results
After training, the network successfully learns the XOR function and produces predictions close to:
- [0, 0] → ~0
- [0, 1] → ~1
- [1, 0] → ~1
- [1, 1] → ~0

## Learning Outcomes
- Understanding of non-linear classification problems
- Implementation of backpropagation algorithm from scratch
- Importance of hidden layers in neural networks
- Weight initialization and gradient descent optimization
- Manual implementation without high-level frameworks

## Requirements
- NumPy

## Usage
Run all cells in sequence to:
1. Define activation functions
2. Set up input data and labels
3. Initialize network parameters
4. Train the network
5. View final predictions

## Historical Context
The XOR problem was famously described by Marvin Minsky and Seymour Papert in 1969, demonstrating the limitations of single-layer perceptrons. The solution using multi-layer networks (like this MLP) was instrumental in the resurgence of neural network research in the 1980s.
