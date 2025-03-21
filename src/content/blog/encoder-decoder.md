---
title: "A Beginner’s Guide to Encoder-Decoder (Seq2Seq) Models"
author: "Md Shahabub Alam"
pubDatetime: 2024-03-01T00:00:00Z
slug: encoder-decoder-model-tutorial
featured: false
draft: false
tags:
  - Machine Learning
  - NLP
  - Seq2Seq
  - Encoder-Decoder
description: "Learn how encoder-decoder (seq2seq) models work with a clear and simple example. This beginner-friendly guide explains the architecture, practical applications, and provides easy-to-follow Python code."
canonicalURL: ""
---

## 📚 Table of Contents

- [Introduction](#introduction)
- [What is an Encoder-Decoder Model?](#what-is-an-encoder-decoder-model)
- [Example Application: Language Translation](#example-application-language-translation)
- [Step-by-Step Tutorial with Python](#step-by-step-tutorial-with-python)
  - [Installing Dependencies](#installing-dependencies)
  - [Preparing Data](#preparing-data)
  - [Building a Simple Seq2Seq Model](#building-a-simple-seq2seq-model)
  - [Training the Model](#training-the-model)
  - [Using the Model for Prediction](#using-the-model-for-prediction)
- [Summary](#summary)
- [References](#references)

---

## 🚀 Introduction

Encoder-decoder models (also known as sequence-to-sequence or Seq2Seq models) have transformed Natural Language Processing (NLP) tasks such as machine translation, text summarization, and question-answering systems. They allow computers to map input sequences (e.g., sentences in one language) to output sequences (translations in another language).

This guide provides a clear, straightforward introduction to how these models work, supported by an easy-to-understand Python example.

---

## 🤖 What is an Encoder-Decoder Model?

A Seq2Seq model consists of two main parts:

- **Encoder**: Processes the input sequence and converts it into a context vector (also known as the hidden state), capturing the meaning of the sequence.
- **Decoder**: Takes this context vector and generates the output sequence, one word at a time.

The key advantage of this architecture is its flexibility in handling sequences of varying lengths and complexity.

---

## 🌐 Example Application: Language Translation

Imagine translating an English sentence into German:

- **Input (English)**: "Hello, how are you?"
- **Output (German)**: "Hallo, wie geht es dir?"

Here, the encoder reads the English sentence, converts it to a context vector, and the decoder generates the German translation from this context.

---

## 🛠️ Step-by-Step Tutorial with Python

Let's create a simplified encoder-decoder model using PyTorch to translate short English phrases to German.

### 📌 Installing Dependencies

Install the required packages:

```bash
pip install torch torchtext numpy
```

```python
data = [
    ("hello", "hallo"),
    ("how are you", "wie geht es dir"),
    ("good morning", "guten morgen"),
    ("thank you", "danke"),
    ("good night", "gute nacht")
]
```

### 📌 Preparing Data

We'll create a tiny dataset manually for simplicity:

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Encoder definition
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        _, hidden = self.gru(embedded, hidden)
        return hidden

# Decoder definition
class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = torch.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden
```


### 📌 Building a Simple Seq2Seq Model
Here's a simple encoder-decoder model in PyTorch:





```python
encoder = Encoder(input_size=10, hidden_size=16)
decoder = Decoder(hidden_size=16, output_size=10)

encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.01)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.01)
criterion = nn.NLLLoss()

# Simplified training loop example
for epoch in range(100):
    input_tensor = torch.tensor([1,2,3])  # dummy inputs
    target_tensor = torch.tensor([1,2,3]) # dummy targets
    encoder_hidden = torch.zeros(1, 1, 16)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    encoder_hidden = encoder(input_tensor[0], encoder_hidden)

    decoder_input = torch.tensor([0])  # start token
    decoder_hidden = encoder_hidden

    loss = 0
    for di in range(target_tensor.size(0)):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        loss += criterion(decoder_output, target_tensor[di].unsqueeze(0))
        decoder_input = target_tensor[di]

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

```

### 📌 Using the Model for Prediction
After training, you can generate translations:

```python
encoder_hidden = torch.zeros(1, 1, 16)
input_tensor = torch.tensor([1,2,3])
encoder_hidden = encoder(input_tensor[0], encoder_hidden)

decoder_input = torch.tensor([0])  # start token
decoder_hidden = encoder_hidden
predicted_sentence = []

for di in range(3):
    decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
    topv, topi = decoder_output.topk(1)
    predicted_sentence.append(topi.item())
    decoder_input = topi.squeeze().detach()

print("Predicted sentence indices:", predicted_sentence)


```

## 🎯 Summary

Encoder-decoder (Seq2Seq) models are powerful tools in NLP tasks like translation. Understanding their structure helps in building effective and versatile NLP solutions.

## 📖 References

- [Sequence to Sequence Learning with Neural Networks (Sutskever et al., 2014)](https://arxiv.org/abs/1409.3215)
- [PyTorch Seq2Seq Tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
- [Understanding Encoder-Decoder Networks for Machine Translation](https://machinelearningmastery.com/encoder-decoder-attention-sequence-to-sequence-prediction-keras/)
