# Character-Level Language Model using MLP

A character-level language model built with a Multi-Layer Perceptron (MLP) that predicts the next character in a sequence given a fixed context window of 3 previous characters. Implemented in PyTorch on a names dataset.


---

## Overview

This model operates at the **character level** — the basic unit of input and output is a single character. Given a context of **3 previous characters**, the model learns to predict the most likely next character.

The model is implemented using a **Multi-Layer Perceptron (MLP)**, which consists of one or more hidden layers with non-linear activations, followed by an output layer that produces a probability distribution over all possible next characters using **softmax**.

---

## Model Architecture

```
Input: 3 characters (context window / block size)
    ↓
Character Embeddings  →  C: (27, 10)  [27 chars, 10-dim embedding]
    ↓
Flatten  →  (3 × 10 = 30 features)
    ↓
Hidden Layer  →  W1: (30, 200) + b1  →  tanh activation
    ↓
Output Layer  →  W2: (200, 27) + b2
    ↓
Softmax  →  Probability distribution over 27 characters
```

| Component         | Details                                    |
|-------------------|--------------------------------------------|
| Vocabulary size   | 27 (a–z + `.` as start/end token)          |
| Context window    | 3 characters                               |
| Embedding size    | 10                                         |
| Hidden layer size | 200 neurons                                |
| Activation        | Tanh                                       |
| Loss function     | Cross-Entropy                              |
| Optimizer         | SGD with decaying learning rate            |
| Training steps    | 200,000                                    |
| Batch size        | 32 (mini-batch)                            |
| Framework         | PyTorch                                    |

---

## Dataset

The model is trained on a dataset of names (`names_makemore.txt`), where each name is a sequence of characters. The `.` token is used as a special start/end-of-word marker.

**Dataset split:**
| Split      | Size  | Purpose                              |
|------------|-------|--------------------------------------|
| Training   | 80%   | Train model weights                  |
| Validation | 10%   | Tune hyperparameters, check overfitting |
| Test       | 10%   | Final evaluation                     |

**Training set size:** 182,625 examples  
**Validation set size:** 22,655 examples  
**Test set size:** 22,866 examples

---

## How It Works

### 1. Vocabulary & Encoding
All unique characters from the dataset are collected and mapped to integer indices. The `.` token (index 0) serves as the start/end-of-word delimiter.

```
{'a': 1, 'b': 2, ..., 'z': 26, '.': 0}
```

### 2. Dataset Construction
Training examples are built using a **sliding window** of size 3. For each word, a context of 3 characters is used to predict the next character.

```
Example: "emma"
... → e
..e → m
.em → m
emm → a
mma → .
```

### 3. Training
- Character indices are passed through a **lookup table (embedding matrix C)** to get dense 10-dimensional vector representations
- The 3 embeddings are **concatenated** into a 30-dimensional vector
- Passed through a **hidden layer** (200 neurons, tanh activation)
- Then through an **output layer** producing 27 logits
- Loss is computed using **cross-entropy**
- Weights are updated via **SGD** with learning rate decay:
  - `lr = 0.1` for the first 100,000 steps
  - `lr = 0.01` for the remaining 100,000 steps

### 4. Evaluation
The model is evaluated on the **validation set** by computing cross-entropy loss to detect overfitting. The final **validation loss is ~2.18**.

### 5. Text Generation
After training, names are generated **autoregressively**:
- Start with a context of 3 `.` tokens
- Predict the next character by sampling from the output probability distribution
- Slide the context window forward and repeat until a `.` token is generated

**Sample generated names:**
```
carpazlaylyn   jari   reety   deliah
nerania        kaleigh   quint   kai
```

---

## Visualizations

The notebook includes two visualizations:

1. **Training loss curve** — plots log10 loss over 200,000 training steps, showing convergence.
2. **Embedding space** — scatter plot of the learned 2D character embeddings (projected from 10D), showing how the model clusters similar characters together.

---

## Getting Started

### Prerequisites

```bash
pip install torch matplotlib jupyter
```

### Run the Notebook

```bash
git clone https://github.com/MaitriVaghela/character-level-language-model-mlp.git
cd character-level-language-model-mlp
git checkout mlp
jupyter notebook language-model-mlp.ipynb
```

> **Note:** The notebook was originally run on Kaggle. To run locally, update the dataset path from `/kaggle/input/datasets/maitrivaghela/namesdata/names_makemore.txt` to your local path.

---

## Project Structure

```
character-level-language-model-mlp/
│
└── language-model-mlp.ipynb   # Full implementation: data prep, training, evaluation, generation
```

---

## Key Concepts

- **Character-level modelling** — operates on raw characters, no tokenizer needed
- **Embeddings** — each character is mapped to a learnable dense vector (10-dim)
- **Context window (block size = 3)** — model uses 3 previous characters to predict the next
- **MLP** — a simple but effective architecture for learning sequential patterns
- **Autoregressive generation** — each predicted character feeds into the next prediction
- **Learning rate decay** — reduces LR mid-training to refine weight updates

---

## References

- [Bengio et al. (2003) — A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)
- [Andrej Karpathy — makemore series](https://github.com/karpathy/makemore)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

## Author

**Maitri Vaghela**  
[GitHub Profile](https://github.com/MaitriVaghela)
