# 🧠 Deep Learning Projects

A comprehensive collection of hands-on deep learning implementations covering fundamental concepts to advanced neural network architectures. Each project is organized in its own folder with a Jupyter notebook and detailed README documentation.

## 📚 Projects

### 1. [XOR using Multi-Layer Perceptron](01-xor-mlp/)
Solving the classic XOR problem with a Multi-Layer Perceptron to demonstrate that neural networks can learn non-linear functions.
- 2-layer neural network implementation from scratch
- Backpropagation algorithm
- Sigmoid activation functions
- Manual gradient descent optimization

**Key Learning**: Understanding non-linear classification, backpropagation, and the importance of hidden layers.

### 2. [Linear Regression](02-linear-regression/)
Implementation of linear regression from scratch using gradient descent optimization without ML libraries.
- Custom gradient descent algorithm
- Mean Squared Error loss function
- Loss curve visualization
- Model evaluation on synthetic data

**Key Learning**: Fundamentals of supervised learning, optimization, and gradient-based learning.

### 3. [Softmax Regression](03-softmax-regression/)
Multi-class classification using Softmax Regression (Multinomial Logistic Regression).
- Softmax activation for probability distribution
- Cross-entropy loss
- One-hot encoding
- Multi-class classification framework

**Key Learning**: Extension from binary to multi-class classification, probabilistic outputs.

### 4. [MNIST CNN Classification](04-mnist-cnn/)
Convolutional Neural Network for handwritten digit recognition on the MNIST dataset.
- Deep CNN architecture with multiple conv layers
- Max pooling and dropout regularization
- Batch normalization
- Achieves ~99% test accuracy

**Key Learning**: CNN fundamentals, convolutional layers, pooling, and computer vision basics.

### 5. [IMDB Sentiment Classification using LSTM](05-imdb-lstm/)
Sentiment analysis on movie reviews using LSTM (Long Short-Term Memory) networks.
- Text preprocessing and tokenization
- Word embeddings
- Stacked LSTM layers
- Binary sentiment classification (positive/negative)

**Key Learning**: Recurrent neural networks, sequence processing, NLP fundamentals, and handling variable-length text.

### 6. [CIFAR-10 Image Classification with InceptionV3](06-cifar10-inceptionv3/)
Transfer learning approach using pre-trained InceptionV3 architecture for CIFAR-10 classification.
- Transfer learning with ImageNet pre-trained weights
- Inception modules for multi-scale feature extraction
- Fine-tuning strategies
- Confusion matrix visualization

**Key Learning**: Transfer learning, advanced CNN architectures, and adapting pre-trained models to new tasks.

### 7. [Multi-Head Attention with PyTorch TransformerEncoder](07-multihead-attention/)
Implementation of the Transformer architecture and multi-head attention mechanism.
- Self-attention mechanism
- Multi-head attention
- Positional encoding
- PyTorch TransformerEncoder implementation

**Key Learning**: Modern NLP architectures, attention mechanisms, and the foundation of BERT/GPT models.

## �️ Project Structure

Each project folder contains:
- **Jupyter Notebook**: Complete implementation with code and outputs
- **README.md**: Detailed documentation explaining:
  - Problem description and motivation
  - Mathematical foundations and algorithms
  - Architecture details
  - Implementation specifics
  - Key concepts and learning outcomes
  - Requirements and usage instructions

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8+
Jupyter Notebook or JupyterLab
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/dlspe-main.git
cd dlspe-main
```

2. Install required packages:
```bash
pip install numpy pandas matplotlib seaborn
pip install tensorflow keras torch torchvision
pip install scikit-learn jupyter tensorflow-datasets
```

3. Navigate to a project folder:
```bash
cd 01-xor-mlp
```

4. Read the README.md for detailed information, then launch Jupyter:
```bash
jupyter notebook
```

## 📊 Results Summary

| Project | Accuracy | Framework | Dataset |
|---------|----------|-----------|---------|
| Linear Regression | MSE: ~1.84 | NumPy | Synthetic |
| Softmax Regression | 100% | NumPy | Iris |
| XOR MLP | 100% | NumPy | XOR |
| MNIST CNN | 99.07% | TensorFlow | MNIST |
| IMDB LSTM | 64.24% | TensorFlow | IMDB |
| CIFAR-10 InceptionV3 | 62% | TensorFlow | CIFAR-10 |
| MultiHead Attention | Varies | PyTorch | Custom |

## 🛠️ Technologies Used

- **NumPy** - Numerical computing and matrix operations
- **TensorFlow/Keras** - Deep learning framework
- **PyTorch** - Neural network implementation
- **Matplotlib/Seaborn** - Data visualization
- **Scikit-learn** - Machine learning utilities

## 📖 Learning Path

Recommended order for beginners:
1. Start with **Linear Regression** to understand optimization
2. Move to **Softmax Regression** for classification
3. Learn neural networks with **XOR using MLP**
4. Explore CNNs with **MNIST CNN**
5. Study RNNs with **IMDB Sentiment Classification**
6. Understand transfer learning with **CIFAR-10**
7. Dive into transformers with **MultiHead Attention**

## 📝 License

**Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**

This work is licensed under a non-commercial license. You are free to:
- ✅ Share and redistribute the material
- ✅ Adapt, remix, and build upon the material
- ✅ Use for educational and personal projects

Under the following terms:
- 📌 **Attribution** - You must give appropriate credit
- ❌ **NonCommercial** - You may NOT use this material for commercial purposes

For commercial use, please contact the author.

[View Full License](LICENSE)

## 👤 Author

**Pradeep Krishna J**
- GitHub: [@PradeepKrishna-J](https://github.com/PradeepKrishna-J)

## 🤝 Contributing

Contributions for educational purposes are welcome! Please feel free to:
- Report bugs
- Suggest improvements
- Add more examples
- Improve documentation

---

⭐ **If you find this helpful, please star the repository!**
