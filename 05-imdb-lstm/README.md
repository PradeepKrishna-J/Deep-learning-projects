# IMDB Sentiment Classification using LSTM

## Overview
This project implements a sentiment analysis system using Long Short-Term Memory (LSTM) networks to classify movie reviews from the IMDB dataset as positive or negative. It demonstrates the power of recurrent neural networks for natural language processing tasks.

## IMDB Dataset

### About the Dataset
- **Source**: Internet Movie Database (IMDB)
- **Task**: Binary sentiment classification
- **Training Samples**: 25,000 movie reviews
- **Test Samples**: 25,000 movie reviews
- **Classes**: Binary (Positive/Negative)
- **Format**: Variable-length text sequences

### Dataset Characteristics
- **Balanced**: Equal positive and negative reviews
- **Polarized**: Only highly positive (≥7/10) or negative (≤4/10) reviews
- **Variable Length**: Reviews range from short sentences to long paragraphs
- **Real-World**: Actual user-written reviews with informal language

## Problem: Text Classification

### Challenges in Text Processing

1. **Variable Length**: Reviews have different lengths
2. **Sequential Nature**: Word order matters ("good, not bad" vs "bad, not good")
3. **Context Dependency**: Meaning depends on surrounding words
4. **Vocabulary Size**: Thousands of unique words
5. **Semantic Understanding**: Requires understanding language nuances

### Traditional vs Deep Learning Approaches

**Traditional Methods**:
- Bag-of-words: Loses word order
- TF-IDF: Better weighting but still no sequence modeling
- N-grams: Limited context window

**Deep Learning (LSTM)**:
- Captures sequential dependencies
- Learns semantic representations
- Handles long-range dependencies
- End-to-end trainable

## Natural Language Processing Pipeline

### 1. Text Preprocessing

#### Tokenization
Converts text to numerical sequences:
```
"This movie was great!" → [12, 45, 78, 234]
```

**Steps**:
- Split text into words
- Build vocabulary (most common words)
- Assign unique integer to each word
- Handle out-of-vocabulary words (OOV)

#### Sequence Padding
Makes all sequences same length:
```
[12, 45, 78, 234] → [12, 45, 78, 234, 0, 0, 0, ...]  (pad to max_length)
```

**Strategies**:
- **Post-padding**: Add zeros at end (common)
- **Pre-padding**: Add zeros at beginning
- **Truncation**: Cut long sequences to max length

### 2. Word Embeddings

#### What are Embeddings?
Transform discrete words into continuous vectors:
```
"movie" → [0.2, -0.5, 0.8, ..., 0.1]  (dense vector)
```

**Purpose**:
- Capture semantic similarity
- Reduce dimensionality (10K vocab → 128D vectors)
- Learn task-specific representations

**Properties**:
- Similar words have similar vectors
- Arithmetic: king - man + woman ≈ queen
- Learned during training or pre-trained (Word2Vec, GloVe)

#### Embedding Layer
```python
Embedding(vocab_size=10000, 
          embedding_dim=128, 
          input_length=200)
```

Converts each word ID to a learnable vector.

## Long Short-Term Memory (LSTM)

### Why RNNs for Text?

**Recurrent Neural Networks (RNNs)**:
- Process sequences step by step
- Maintain hidden state (memory)
- Share parameters across time steps

**Problem with vanilla RNNs**:
- **Vanishing gradients**: Can't learn long-range dependencies
- **Exploding gradients**: Training instability

### LSTM Architecture

LSTMs solve RNN problems through a gating mechanism:

#### 1. **Forget Gate**
Decides what information to discard from cell state:
```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
```

#### 2. **Input Gate**
Decides what new information to store:
```
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)
```

#### 3. **Cell State Update**
Updates the long-term memory:
```
C_t = f_t * C_{t-1} + i_t * C̃_t
```

#### 4. **Output Gate**
Decides what to output:
```
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
h_t = o_t * tanh(C_t)
```

### Why LSTM Works

1. **Cell State**: Highway for information flow
2. **Gates**: Learn what to remember/forget
3. **No Gradient Vanishing**: Direct path through cell state
4. **Selective Memory**: Keeps relevant information, forgets irrelevant

### LSTM Variants

#### Stacked LSTM
Multiple LSTM layers for hierarchical learning:
```python
LSTM(64, return_sequences=True)  # First layer
LSTM(64)                          # Second layer
```

#### Bidirectional LSTM
Processes sequence forward and backward:
- Captures future context
- Better understanding of word meaning
- Double the parameters

## Model Architecture

### Typical Sentiment Analysis Architecture
```
Input (sequence of word IDs)
    ↓
Embedding Layer (vocab_size → embedding_dim)
    ↓
LSTM Layer 1 (return_sequences=True)
    ↓
Dropout (regularization)
    ↓
LSTM Layer 2
    ↓
Dense Layer (ReLU activation)
    ↓
Dropout
    ↓
Output Layer (Sigmoid) → [0, 1]
```

### Layer-by-Layer Explanation

1. **Embedding**: Words → Vectors (10000 → 128D)
2. **LSTM 1**: Learn sequential patterns, output for each timestep
3. **Dropout**: Prevent overfitting (20% neurons dropped)
4. **LSTM 2**: Higher-level sequential patterns, final state only
5. **Dense**: Non-linear transformation (64 units)
6. **Dropout**: More regularization (20%)
7. **Output**: Binary classification (Sigmoid)

## Training Process

### 1. Loss Function
**Binary Cross-Entropy**:
```
Loss = -[y·log(ŷ) + (1-y)·log(1-ŷ)]
```

Appropriate for binary classification.

### 2. Optimizer
**Adam** (Adaptive Moment Estimation):
- Adaptive learning rates per parameter
- Combines momentum and RMSprop
- Fast convergence
- Default: lr=0.001

### 3. Training Configuration
```python
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          epochs=5-10,
          batch_size=32,
          validation_split=0.2)
```

### 4. Regularization Techniques

**Dropout**: Randomly drop neurons during training
- Prevents co-adaptation
- Typical rates: 0.2-0.5

**Early Stopping**: Stop when validation loss stops improving

**Gradient Clipping**: Prevent exploding gradients

## Hyperparameters

### Critical Parameters

1. **Vocabulary Size**: 10000 (balance coverage vs computation)
2. **Embedding Dimension**: 128 (semantic representation quality)
3. **Max Sequence Length**: 200-300 (longer captures more context)
4. **LSTM Units**: 64-128 (model capacity)
5. **Dropout Rate**: 0.2-0.5 (regularization strength)
6. **Batch Size**: 32-128 (training stability)
7. **Learning Rate**: 0.001 (Adam default, usually works well)

## Evaluation and Results

### Metrics
1. **Accuracy**: Overall correctness
2. **Precision/Recall**: Per-class performance
3. **ROC-AUC**: Threshold-independent performance
4. **Confusion Matrix**: Error analysis

### Expected Performance
- **Baseline (Bag-of-Words + Logistic)**: ~85-88%
- **LSTM**: ~88-90%
- **Bidirectional LSTM**: ~89-91%
- **BERT (Transformer)**: ~93-95%

### Inference Example
```python
predict_sentiment("This movie was amazing!")  # → Positive
predict_sentiment("Terrible waste of time")   # → Negative
```

## Challenges and Solutions

### 1. Training Time
**Problem**: LSTMs are slow to train
**Solutions**:
- Use GPU acceleration
- Reduce sequence length
- Reduce vocabulary size
- Use simpler models for prototyping

### 2. Overfitting
**Problem**: Model memorizes training data
**Solutions**:
- Increase dropout
- Reduce model size
- Early stopping
- More training data

### 3. Gradient Issues
**Problem**: Exploding/vanishing gradients
**Solutions**:
- Gradient clipping (for exploding)
- Use LSTM instead of vanilla RNN (for vanishing)
- Batch normalization

## Advanced Techniques

### 1. Attention Mechanisms
- Focus on important words
- Visualize what model attends to
- Improves interpretability

### 2. Pre-trained Embeddings
- Word2Vec, GloVe, FastText
- Transfer learning from large corpora
- Faster convergence

### 3. Transformer Models
- BERT, RoBERTa, GPT
- State-of-the-art performance
- Bi-directional context
- Pre-trained on massive datasets

### 4. Ensemble Methods
- Combine multiple models
- Reduce variance
- Improve robustness

## Requirements
- TensorFlow or PyTorch
- TensorFlow Datasets (for IMDB data)
- NumPy
- Matplotlib (optional, for visualization)

## Real-World Applications

Similar architectures are used for:
- **Product Review Analysis**: E-commerce sentiment
- **Social Media Monitoring**: Brand sentiment tracking
- **Customer Support**: Auto-categorize tickets
- **Financial News**: Market sentiment analysis
- **Political Analysis**: Opinion mining
- **Content Moderation**: Detect toxic comments

## Learning Outcomes

After completing this project:
- Understand text preprocessing pipeline
- Implement word embeddings
- Grasp LSTM architecture and purpose
- Handle variable-length sequences
- Apply deep learning to NLP
- Evaluate text classification models
- Foundation for modern NLP (Transformers)

## From LSTM to Transformers

**Evolution of NLP Models**:
1. **Bag-of-Words**: No sequence modeling
2. **RNNs**: Basic sequential processing
3. **LSTMs**: Better long-range dependencies
4. **Attention**: Focus on relevant parts
5. **Transformers**: State-of-the-art (BERT, GPT)

LSTMs remain valuable for:
- Understanding sequence modeling
- Resource-constrained environments
- Streaming/online processing
- Teaching fundamental concepts

## Best Practices

1. **Start Simple**: Begin with small vocab and short sequences
2. **Monitor Validation**: Catch overfitting early
3. **Experiment with Hyperparameters**: Grid/random search
4. **Visualize Predictions**: Understand failure cases
5. **Use Pre-trained Embeddings**: When possible
6. **Regularize Heavily**: NLP models prone to overfitting
7. **Analyze Errors**: Learn from misclassifications
