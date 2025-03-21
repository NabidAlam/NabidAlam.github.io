---
title: "Building a Transformer from Scratch with PyTorch"
pubDatetime: 2024-01-16T00:00:00Z
description: "Learn how the Transformer model works and how to implement it from scratch in PyTorch. This guide covers key components like multi-head attention, positional encoding, and training."
slug: "transformer-from-scratch"
featured: false
draft: false
tags:
  - Machine Learning
  - NLP
  - Transformer
  - Deep Learning
canonicalURL: ""
---

## 📚 Table of Contents

- [Introduction](#introduction)
- [Why Do We Need Transformers?](#why-do-we-need-transformers)
- [How Transformers Work](#how-transformers-work)
- [Mathematical Formulation](#mathematical-formulation)
- [Step-by-Step Transformer Implementation](#step-by-step-transformer-implementation)
  - [Installing Dependencies](#installing-dependencies)
  - [Positional Encoding](#positional-encoding)
  - [Multi-Head Self-Attention](#multi-head-self-attention)
  - [Feed-Forward Layer](#feed-forward-layer)
  - [Building the Transformer Encoder](#building-the-transformer-encoder)
  - [Building the Transformer Decoder](#building-the-transformer-decoder)
  - [Putting It All Together](#putting-it-all-together)
- [Training the Transformer](#training-the-transformer)
- [Conclusion](#conclusion)
- [References](#references)

---

## 🚀 Introduction

The **Transformer model** has revolutionized NLP by replacing RNNs with a fully attention-based architecture. Introduced by Vaswani et al. in 2017, it powers models like **GPT, BERT, and T5**.

Unlike RNNs, Transformers process entire sequences **in parallel**, making them faster and more efficient for tasks like **translation, summarization, and text generation**.

---

## ❓ Why Do We Need Transformers?

- **No Sequential Bottleneck** → Unlike RNNs, Transformers process words **simultaneously**, enabling parallel computation.
- **Better Long-Range Dependencies** → Attention mechanisms allow **direct connections** between distant words.
- **Scalability** → Transformers scale **efficiently** with large datasets.

---

## 🔍 How Transformers Work

The Transformer consists of:

1. **Encoder**: Processes the input sequence into a rich **context representation**.
2. **Decoder**: Generates the output sequence step-by-step, attending to the encoded context.

Each layer in both the **Encoder and Decoder** contains:
- **Multi-Head Self-Attention**
- **Feed-Forward Layers**
- **Add & Norm Layers (Residual Connections + LayerNorm)**

---

## 🛠️ Step-by-Step Transformer Implementation

### 📌 Installing Dependencies

```bash
pip install torch numpy matplotlib
```

---

### 🔧 **Positional Encoding**
```python
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
```

---

### 🔧 **Multi-Head Self-Attention**
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, Q, K, V):
        Q = self.W_q(Q).view(Q.shape[0], Q.shape[1], self.num_heads, self.head_dim).transpose(1,2)
        K = self.W_k(K).view(K.shape[0], K.shape[1], self.num_heads, self.head_dim).transpose(1,2)
        V = self.W_v(V).view(V.shape[0], V.shape[1], self.num_heads, self.head_dim).transpose(1,2)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        output = torch.matmul(attention_weights, V).transpose(1, 2).reshape(Q.shape[0], -1, self.num_heads * self.head_dim)

        return self.fc_out(output)
```

---

### 🔧 **Building the Transformer Encoder**
```python
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim):
        super(TransformerEncoderLayer, self).__init__()
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, ffn_dim), nn.ReLU(), nn.Linear(ffn_dim, embed_dim))
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.norm1(x + self.attn(x, x, x))  # Self-attention + residual connection
        x = self.norm2(x + self.ffn(x))  # Feed-forward + residual connection
        return x
```

---

### 🔧 **Building the Transformer Decoder**
```python
class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.enc_dec_attn = MultiHeadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, ffn_dim), nn.ReLU(), nn.Linear(ffn_dim, embed_dim))
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)

    def forward(self, x, encoder_output):
        x = self.norm1(x + self.self_attn(x, x, x))
        x = self.norm2(x + self.enc_dec_attn(x, encoder_output, encoder_output))
        x = self.norm3(x + self.ffn(x))
        return x
```

---

### 🔧 **Putting It All Together**
```python
class Transformer(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim):
        super(Transformer, self).__init__()
        self.encoder = TransformerEncoderLayer(embed_dim, num_heads, ffn_dim)
        self.decoder = TransformerDecoderLayer(embed_dim, num_heads, ffn_dim)

    def forward(self, src, tgt):
        enc_output = self.encoder(src)
        dec_output = self.decoder(tgt, enc_output)
        return dec_output
```

---

### 📌 **Training the Transformer**
```python
model = Transformer(embed_dim=512, num_heads=8, ffn_dim=2048)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(10):
    src = torch.randn(1, 10, 512)  # Simulated input
    tgt = torch.randn(1, 10, 512)  # Simulated output
    output = model(src, tgt)
    loss = output.sum()  # Dummy loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

### 🎯 **Conclusion**

- **Transformers** use self-attention to process entire sequences efficiently.
- **Multi-Head Attention** allows multiple perspectives on input sequences.
- **Positional Encoding** provides sequence order information.
- **Layer normalization and residual connections** help stabilize training.
- **PyTorch provides a built-in `nn.Transformer` module**, but implementing it from scratch deepens understanding.

By implementing a Transformer from scratch, you've built the foundation of models like **GPT, BERT, and T5**. The next step is fine-tuning on real-world tasks such as **machine translation, text summarization, and chatbot development**.

---

### 📖 **References**

- Vaswani et al., 2017 - ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762)
- Jay Alammar - ["The Illustrated Transformer"](https://jalammar.github.io/illustrated-transformer/)
- PyTorch Documentation - ["Transformer Module"](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html)
- Sebastian Ruder - ["An Overview of Self-Attention Mechanisms"](https://ruder.io/self-attention/)
- Yannic Kilcher - ["Deep Dive into Transformers"](https://www.youtube.com/watch?v=px1mPzVOIb8)
