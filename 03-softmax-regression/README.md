# Softmax Regression (Multinomial Logistic Regression)

## Overview
This project implements Softmax Regression, also known as Multinomial Logistic Regression, which extends binary logistic regression to handle multi-class classification problems. It's commonly used for classifying data into more than two categories.

## Problem Description
Softmax regression is a generalization of logistic regression for multi-class classification where:
- **Input**: Feature vector X
- **Output**: Probability distribution over K classes
- **Goal**: Predict the most likely class for given input

Unlike one-vs-all classification, softmax regression directly models the probability distribution across all classes simultaneously.

## Mathematical Foundation

### 1. Softmax Function
The softmax function converts raw scores (logits) into probabilities:

**P(y=k|X) = exp(z_k) / Σ(exp(z_j))** for all j

Where:
- z_k = X·W_k + b_k (linear combination for class k)
- The output is a probability distribution (sums to 1)
- Each class gets a probability between 0 and 1

### 2. Cross-Entropy Loss
The loss function for softmax regression:

**Loss = -Σ y_true·log(y_pred)**

Or for a single example:
**L = -log(P(y=correct_class|X))**

This loss:
- Is convex (has a single global minimum)
- Penalizes confident wrong predictions heavily
- Is differentiable for gradient-based optimization

### 3. Gradient Computation
The gradient of the loss w.r.t. weights:

**∂L/∂W = X^T·(y_pred - y_true)**

This elegant form makes training efficient.

## Architecture

### Model Structure
- **Input**: Feature vector (e.g., 784 dimensions for MNIST)
- **Linear Layer**: W·X + b
- **Softmax Activation**: Converts logits to probabilities
- **Output**: Probability distribution over classes

### No Hidden Layers
Softmax regression is a **linear model** with no hidden layers, making it:
- Simple and interpretable
- Fast to train
- Suitable for linearly separable data
- A baseline for more complex models

## Typical Applications

1. **MNIST Digit Classification**: Classifying handwritten digits (0-9)
2. **Document Categorization**: Classifying text into topics
3. **Image Classification**: Basic image recognition tasks
4. **Sentiment Analysis**: Multi-class sentiment (negative/neutral/positive)

## Implementation Components

### 1. Data Preprocessing
- Normalization: Scale features to [0, 1] or standardize
- One-hot encoding: Convert labels to categorical format
- Train/test split: Separate data for evaluation

### 2. Model Initialization
- **Weights**: Initialize to small random values
- **Biases**: Usually initialize to zeros
- Shape: [num_features, num_classes]

### 3. Training Loop
```
For each epoch:
    1. Forward pass: compute logits → apply softmax → get probabilities
    2. Compute cross-entropy loss
    3. Backward pass: compute gradients
    4. Update parameters: W = W - α·∇W
    5. Log metrics (loss, accuracy)
```

### 4. Prediction
```
1. Compute probabilities: softmax(W·X + b)
2. Select class with highest probability: argmax(probabilities)
```

## Hyperparameters

Typical hyperparameters to tune:
- **Learning Rate**: Controls step size (e.g., 0.01, 0.1)
- **Batch Size**: Number of samples per update (e.g., 32, 64, 128)
- **Epochs**: Number of passes through dataset (e.g., 10-50)
- **Regularization**: L2 penalty to prevent overfitting

## Advantages

1. **Probabilistic Interpretation**: Outputs are calibrated probabilities
2. **Multi-class Ready**: Naturally handles multiple classes
3. **Convex Optimization**: Guaranteed to find global minimum
4. **Efficient**: Fast training and inference
5. **Interpretable**: Weights show feature importance per class

## Limitations

1. **Linear Decision Boundaries**: Cannot model complex non-linear patterns
2. **Feature Engineering**: Requires good features for complex problems
3. **Not State-of-the-Art**: Deep neural networks perform better on complex tasks
4. **Assumes Independence**: Treats features as independent

## Comparison with Other Models

| Model | Decision Boundary | Complexity | Use Case |
|-------|------------------|------------|----------|
| Softmax Regression | Linear | Low | Simple multi-class problems |
| Neural Networks | Non-linear | High | Complex patterns, images |
| Decision Trees | Non-linear | Medium | Interpretable, tabular data |
| SVM | Linear/Non-linear | Medium | Binary/small multi-class |

## Common Datasets

1. **MNIST**: Handwritten digit classification (10 classes)
2. **Fashion-MNIST**: Clothing item classification (10 classes)
3. **CIFAR-10**: Natural images (10 classes)
4. **Iris**: Flower species (3 classes)

## Evaluation Metrics

1. **Accuracy**: Percentage of correct predictions
2. **Cross-Entropy Loss**: Measures prediction confidence
3. **Confusion Matrix**: Shows per-class performance
4. **Per-Class Precision/Recall**: Identifies class-specific issues

## Extensions and Improvements

1. **L2 Regularization (Ridge)**: Add λ·||W||² to loss
2. **L1 Regularization (Lasso)**: Add λ·||W||₁ to loss
3. **Mini-batch Training**: Use batches for faster convergence
4. **Learning Rate Scheduling**: Decrease learning rate over time
5. **Feature Engineering**: Create polynomial or interaction features

## Requirements
- NumPy (numerical computations)
- Matplotlib (visualization)
- TensorFlow/PyTorch or NumPy (depending on implementation)

## Learning Outcomes

After completing this project, you will understand:
- Softmax activation and its properties
- Cross-entropy loss for classification
- Gradient descent for multi-class problems
- Difference between binary and multi-class classification
- Importance of linear baselines
- Probability calibration and interpretation
- Foundation for neural networks

## From Softmax to Neural Networks

Softmax regression is the **output layer** of most classification neural networks:
- Neural network = Feature learning layers + Softmax layer
- Deep learning extends softmax regression with learned features
- Understanding softmax is crucial for understanding modern deep learning

## Best Practices

1. **Normalize inputs**: Speeds up convergence
2. **Use vectorization**: Much faster than loops
3. **Monitor both loss and accuracy**: Catch overfitting
4. **Try different learning rates**: Critical hyperparameter
5. **Use mini-batches**: Balance speed and convergence
6. **Initialize weights carefully**: Too large causes divergence
7. **Check gradient numerically**: Verify implementation
