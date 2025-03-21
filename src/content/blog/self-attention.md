---
title: "Self-Attention in Neural Networks: A Simple Guide with Examples"
pubDatetime: 2025-03-21T00:00:00Z
description: "Learn how self-attention works in neural networks, particularly in Transformers. This beginner-friendly guide explains the concept with an intuitive example and PyTorch implementation."
slug: "self-attention-tutorial"
featured: false
draft: false
tags:
  - Machine Learning
  - NLP
  - Self-Attention
  - Transformers
canonicalURL: ""
---

## 📚 Table of Contents

- [Introduction](#introduction)
- [Why Do We Need Self-Attention?](#why-do-we-need-self-attention)
- [How Self-Attention Works](#how-self-attention-works)
- [Mathematical Breakdown](#mathematical-breakdown)
- [Step-by-Step Example with Python](#step-by-step-example-with-python)
  - [Installing Dependencies](#installing-dependencies)
  - [Implementing Self-Attention](#implementing-self-attention)
  - [Running the Code](#running-the-code)
- [Summary](#summary)
- [References](#references)

---

## 🚀 Introduction
Self-attention is a key mechanism in deep learning models, especially Transformers, which allows neural networks to weigh different parts of an input sequence when making predictions. Unlike traditional attention, self-attention works **within** a sequence, meaning each token attends to all others.
---

## ❓ Why Do We Need Self-Attention?

- **Capturing Long-Range Dependencies:** Self-attention allows a model to focus on important words, no matter how far apart they are.
- **Parallel Processing:** Unlike RNNs, which process tokens sequentially, self-attention can process all tokens **simultaneously**, making it much faster.
- **Foundation of Transformers:** The Transformer architecture (used in GPT, BERT, etc.) is built entirely on self-attention, replacing RNNs and CNNs in NLP.

---

## 🔍 How Self-Attention Works

Given an input sentence, self-attention assigns a score to each word based on its relevance to other words. The process involves:

1. **Create Query (Q), Key (K), and Value (V) Matrices**
   - Each input token is **projected** into three different vectors.

2. **Compute Attention Scores**
   - Scores are computed using the formula:

     $$
     \text{Attention} = \text{softmax} \left( \frac{Q K^T}{\sqrt{d_k}} \right) V
     $$

   - This determines how much focus each word should get.

3. **Generate the Output**
   - The weighted sum of values forms the new representation of each word.


---

## 🧮 Mathematical Breakdown

The self-attention mechanism follows these key steps:

1. **Compute Query (Q), Key (K), and Value (V) Matrices**:  
   $$
   Q = X W_Q, \quad K = X W_K, \quad V = X W_V
   $$
   where \( W_Q, W_K, W_V \) are learned weight matrices.

2. **Compute Attention Scores**:  
   $$
   \text{Scores} = \frac{Q K^T}{\sqrt{d_k}}
   $$

3. **Apply Softmax to Normalize Scores**:  
   $$
   \alpha = \text{softmax}(\text{Scores})
   $$

4. **Compute Final Self-Attention Output**:  
   $$
   \text{Output} = \alpha V
   $$

---

## 🛠️ Step-by-Step Example with Python

### 📌 Installing Dependencies

```bash
pip install torch numpy
```

---

### 🔧 Implementing Self-Attention in PyTorch

```python
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.W_q = nn.Linear(embed_size, embed_size, bias=False)
        self.W_k = nn.Linear(embed_size, embed_size, bias=False)
        self.W_v = nn.Linear(embed_size, embed_size, bias=False)
        self.scale = torch.sqrt(torch.tensor(embed_size, dtype=torch.float32))

    def forward(self, x):
        Q = self.W_q(x)  # Query matrix
        K = self.W_k(x)  # Key matrix
        V = self.W_v(x)  # Value matrix

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(attention_weights, V)
        return output, attention_weights
```

---

### ➕ Running the Code

```python
embed_size = 8  # Example embedding size
sequence_length = 5  # Example sequence length
batch_size = 1

# Random input tensor (batch_size, sequence_length, embed_size)
x = torch.randn(batch_size, sequence_length, embed_size)

# Initialize self-attention layer and pass the input
self_attention = SelfAttention(embed_size)
output, attention_weights = self_attention(x)

print("Self-Attention Output:\n", output)
print("Attention Weights:\n", attention_weights)
```

This implementation processes a sequence of **5 words**, each represented by an **8-dimensional embedding**. The output represents **contextualized word embeddings**, where each word’s representation depends on other words in the sequence.

---

## 🎯 Summary

- **Self-Attention enables models to weigh different parts of a sequence dynamically.**
- **It replaces recurrence (RNNs) and convolution (CNNs) in NLP models.**
- **Transformers like GPT and BERT use self-attention to capture long-range dependencies.**

---

## 📖 References

- [Vaswani et al., 2017 - Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [The Illustrated Transformer by Jay Alammar](https://jalammar.github.io/illustrated-transformer/)
- [Self-Attention in PyTorch (Official Tutorial)](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)

