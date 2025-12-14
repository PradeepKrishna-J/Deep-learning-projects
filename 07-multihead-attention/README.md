# Multi-Head Attention with PyTorch TransformerEncoder

## Overview
This project implements Multi-Head Attention and PyTorch's TransformerEncoder, demonstrating the core mechanism behind modern NLP breakthroughs like BERT, GPT, and other Transformer-based models. It showcases how attention mechanisms revolutionized deep learning.

## The Transformer Revolution

### Why Transformers?

**Before Transformers (RNNs/LSTMs)**:
- ❌ Sequential processing (slow, hard to parallelize)
- ❌ Limited context (vanishing gradients for long sequences)
- ❌ Fixed-size memory bottleneck
- ✓ Good at local dependencies

**After Transformers**:
- ✓ Parallel processing (much faster training)
- ✓ Unlimited context (attention to entire sequence)
- ✓ Direct connections between all positions
- ✓ Scales to very large models (GPT-3, GPT-4)

### Key Papers
1. **"Attention is All You Need" (2017)**: Original Transformer
2. **"BERT" (2018)**: Bidirectional Transformer for NLP
3. **"GPT-2/3" (2019/2020)**: Generative pre-training at scale
4. **"Vision Transformer" (2020)**: Transformers for images

## Attention Mechanism

### Intuition

**Problem**: How to focus on relevant parts of input?

**Example**:
```
Sentence: "The cat sat on the mat because it was comfortable"
Question: What was comfortable?
Answer: The mat (attention focuses on "mat")
```

Traditional models process sequentially. Attention allows direct connections to relevant information.

### Self-Attention: Core Idea

**Self-Attention** computes relationships between all words in a sequence simultaneously:

1. Each word looks at every other word
2. Determines how much to "attend" to each word
3. Creates context-aware representations

**Formula**:
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

Where:
- **Q**: Query (what am I looking for?)
- **K**: Key (what do I contain?)
- **V**: Value (what do I actually communicate?)

### Step-by-Step Example

**Input**: "The cat sat"

**Step 1: Create Q, K, V vectors**
```
Each word → Linear projection → Q, K, V
"The" → Q_the, K_the, V_the
"cat" → Q_cat, K_cat, V_cat
"sat" → Q_sat, K_sat, V_sat
```

**Step 2: Compute attention scores**
```
Score(cat, The) = Q_cat · K_The
Score(cat, cat) = Q_cat · K_cat
Score(cat, sat) = Q_cat · K_sat
```

**Step 3: Normalize with softmax**
```
Attention_weights = softmax([Score(cat,The), Score(cat,cat), Score(cat,sat)])
Example: [0.1, 0.7, 0.2]  (cat attends mostly to itself)
```

**Step 4: Weighted sum of values**
```
Output_cat = 0.1*V_The + 0.7*V_cat + 0.2*V_sat
```

### Why Scaling Factor √d_k?

Without scaling, dot products grow large with dimension:
- Large dot products → extreme softmax values
- Gradients become very small
- Training instability

**Solution**: Divide by √d_k (square root of key dimension)

## Multi-Head Attention

### Why Multiple Heads?

**Single attention head**: Captures one type of relationship

**Multiple heads**: Capture different relationships simultaneously

**Example**:
- **Head 1**: Subject-verb agreement ("cat" ↔ "sat")
- **Head 2**: Modifier relationships ("black" ↔ "cat")
- **Head 3**: Long-range dependencies ("because" ↔ "comfortable")

### Multi-Head Attention Formula

```
MultiHead(Q, K, V) = Concat(head₁, ..., head_h) W^O

where head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)
```

**Process**:
1. Split Q, K, V into h heads
2. Each head performs attention independently
3. Concatenate all head outputs
4. Linear projection to original dimension

### Configuration

Typical configuration (BERT-base):
- **Number of heads**: 8 or 12
- **Model dimension**: 768
- **Head dimension**: 768 / 12 = 64
- Each head has its own Q, K, V projection matrices

## Transformer Encoder

### Architecture

```
Input Sequence
    ↓
Input Embedding + Positional Encoding
    ↓
┌─────────────────────────────────┐
│  Transformer Encoder Block      │
│  ┌──────────────────────────┐  │
│  │ Multi-Head Attention     │  │
│  └──────────────────────────┘  │
│           ↓                     │
│      Add & Normalize            │
│           ↓                     │
│  ┌──────────────────────────┐  │
│  │ Feed-Forward Network     │  │
│  └──────────────────────────┘  │
│           ↓                     │
│      Add & Normalize            │
└─────────────────────────────────┘
           ↓
    (Repeat N times)
           ↓
    Output Sequence
```

### Components Breakdown

#### 1. Input Embeddings
**Purpose**: Convert tokens to dense vectors

```python
Embedding(vocab_size, d_model)
```

Each word → learnable vector of dimension d_model

#### 2. Positional Encoding
**Problem**: Attention has no notion of position/order

**Solution**: Add positional information
```python
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Properties**:
- Unique for each position
- Allows model to learn relative positions
- Sinusoidal patterns with different frequencies

#### 3. Multi-Head Attention
(Explained above)

#### 4. Feed-Forward Network (FFN)
**Purpose**: Add non-linearity and capacity

```python
FFN(x) = ReLU(xW₁ + b₁)W₂ + b₂
```

**Characteristics**:
- Applied to each position independently
- Typically: d_model → 4*d_model → d_model
- Same across all positions (parameter sharing)

#### 5. Layer Normalization
**Purpose**: Stabilize training

```python
LayerNorm(x) = γ * (x - μ) / (σ + ε) + β
```

**Benefits**:
- Reduces internal covariate shift
- Enables deeper networks
- Faster convergence

#### 6. Residual Connections
**Purpose**: Enable gradient flow in deep networks

```python
output = LayerNorm(x + Sublayer(x))
```

**Benefits**:
- Gradients flow directly through network
- Easier to train very deep models
- Each layer learns refinements, not full transformation

### Stacking Encoder Blocks

**BERT-base**: 12 encoder layers
**BERT-large**: 24 encoder layers
**GPT-3**: 96 decoder layers

More layers → More capacity → Better representations

## PyTorch Implementation

### Basic Transformer Encoder

```python
import torch.nn as nn

# Define encoder layer
encoder_layer = nn.TransformerEncoderLayer(
    d_model=512,        # Embedding dimension
    nhead=8,            # Number of attention heads
    dim_feedforward=2048, # FFN dimension
    dropout=0.1,        # Dropout probability
    activation='relu'   # FFN activation
)

# Stack multiple layers
transformer_encoder = nn.TransformerEncoder(
    encoder_layer,
    num_layers=6        # Number of encoder blocks
)

# Forward pass
output = transformer_encoder(src, src_key_padding_mask=padding_mask)
```

### Key Parameters

- **d_model**: Model dimension (must be divisible by nhead)
- **nhead**: Number of attention heads
- **dim_feedforward**: Inner FFN dimension (usually 4×d_model)
- **dropout**: Regularization (typical: 0.1-0.3)
- **num_layers**: Depth of the encoder

## Masking

### Why Masking?

1. **Padding Mask**: Ignore padding tokens
2. **Attention Mask**: Control which positions can attend to which

### Padding Mask
```python
# Sequence: ["Hello", "world", "<PAD>", "<PAD>"]
padding_mask = [False, False, True, True]
```

Prevents attention to padding tokens.

### Causal Mask (for Decoders)
```python
# Upper triangular mask
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
```

Prevents attention to future tokens (for autoregressive generation).

## Training Considerations

### 1. Positional Encoding
Must be added before feeding to encoder:
```python
x = embeddings + positional_encoding
```

### 2. Sequence Length
- Fixed during training (padding/truncation)
- Typical: 128, 256, 512 tokens
- Longer sequences → more memory, slower

### 3. Batch First
PyTorch default: (seq_len, batch, d_model)
Set `batch_first=True` for (batch, seq_len, d_model)

### 4. Gradient Accumulation
For large models:
```python
for i, batch in enumerate(dataloader):
    loss = model(batch)
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

## Applications

### 1. Text Classification
```
[CLS] token → Transformer → Take [CLS] output → Classifier
```

### 2. Sequence Labeling (NER, POS)
```
Tokens → Transformer → Output for each token → Tag each word
```

### 3. Question Answering
```
[CLS] Question [SEP] Context → Transformer → Predict answer span
```

### 4. Language Modeling
```
Tokens → Transformer → Predict next token
```

### 5. Machine Translation
```
Source → Transformer Encoder → Decoder → Target
```

## Comparison with Other Architectures

| Feature | RNN/LSTM | CNN | Transformer |
|---------|----------|-----|-------------|
| Parallelization | ❌ Sequential | ✓ Fully parallel | ✓ Fully parallel |
| Long-range dependencies | ⚠️ Limited | ❌ Very limited | ✓ Direct |
| Computational complexity | O(n) | O(1) | O(n²) |
| Memory usage | Low | Low | High |
| Position awareness | Implicit | Via convolution | Via positional encoding |

## Advantages of Transformers

1. **Parallelization**: Process entire sequence at once
2. **Long-range dependencies**: Direct paths between all positions
3. **Interpretability**: Attention weights show what model focuses on
4. **Transfer learning**: Pre-trained models (BERT, GPT) work extremely well
5. **Flexibility**: Works for text, images, audio, video
6. **Scalability**: Scales to billions of parameters

## Limitations

1. **Quadratic complexity**: O(n²) in sequence length
   - Solutions: Sparse attention, Linformer, Performer
2. **No inductive bias**: Needs more data than CNNs for images
3. **Positional encoding**: Not as natural as RNNs
4. **Memory intensive**: Large models require significant GPU memory

## Modern Variants

### Efficient Transformers
- **Longformer**: Attention for long documents
- **Reformer**: Locality-sensitive hashing
- **Linformer**: Linear complexity attention

### Vision Transformers
- **ViT**: Pure transformer for images (no convolutions)
- **DeiT**: Data-efficient image transformers
- **Swin Transformer**: Hierarchical vision transformer

### Other Domains
- **Music Transformer**: Music generation
- **AlphaFold**: Protein structure prediction
- **Decision Transformer**: Reinforcement learning

## Requirements
- PyTorch
- NumPy
- Matplotlib (visualization)
- Optional: Transformers library (Hugging Face)

## Learning Outcomes

After completing this project:
- Understand attention mechanism mathematically
- Implement multi-head attention
- Grasp transformer encoder architecture
- Use PyTorch's TransformerEncoder
- Handle masking for padding and causality
- Apply positional encoding
- Understand residual connections and layer normalization
- Foundation for modern NLP (BERT, GPT)
- Basis for Vision Transformers
- Understand trade-offs vs RNNs/CNNs

## From Transformers to GPT/BERT

**BERT** (Bidirectional Encoder):
- Stack of Transformer encoders
- Pre-trained on masked language modeling
- Fine-tuned for downstream tasks

**GPT** (Generative Pre-trained Transformer):
- Stack of Transformer decoders (causal attention)
- Pre-trained on next token prediction
- Few-shot learning capabilities

**This project**: Foundation for understanding these models!

## Best Practices

1. **Start Small**: Use small d_model and few layers initially
2. **Visualize Attention**: Understand what model learns
3. **Use Pre-trained Models**: Leverage Hugging Face Transformers
4. **Monitor Training**: Watch for attention collapse
5. **Experiment with Heads**: Different numbers capture different patterns
6. **Proper Normalization**: Critical for training stability
7. **Learning Rate Warmup**: Helps transformer training
8. **Gradient Clipping**: Prevent exploding gradients

## Resources

- **Paper**: "Attention is All You Need" (Vaswani et al., 2017)
- **The Illustrated Transformer**: Visual explanation
- **Hugging Face**: Pre-trained transformer models
- **Annotated Transformer**: Line-by-line implementation guide
