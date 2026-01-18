# Natural-Language-Processing_Autumn_2025

## Project Overview

This repository contains the practical assignments and implementations associated with the Natural Language Processing (NLP) Bootcamp. The primary objective of this project is to explore and implement computational linguistics techniques and deep learning architectures for text classification tasks using Python and the PyTorch framework.

The coursework progresses from fundamental text processing and traditional machine learning models to advanced neural network architectures. It provides a comparative analysis of their performance on sentiment analysis tasks. The project demonstrates the complete NLP pipeline, including text preprocessing, vectorization, model architecture design, training loops, and evaluation metrics.

## Repository Contents

The project is structured into two main assignments:

1.  **Assignment 1: Text Preprocessing and Baseline Classification**
    * Focuses on the fundamentals of Natural Language Understanding (NLU) using the Natural Language Toolkit (NLTK).
    * Implements traditional machine learning algorithms (Logistic Regression) as a performance baseline.
    * Explores semantic representation using Word Embeddings (Word2Vec).
2.  **Assignment 2: Deep Learning for Text Classification**
    * Focuses on building neural network architectures using PyTorch.
    * Implements Feedforward Neural Networks (Artificial Neural Networks) for text.
    * Implements Recurrent Neural Networks (RNN) to handle sequential data and capture temporal dependencies.

## Technical Prerequisites

To execute the code contained in this repository, the following software and libraries are required:

* **Python 3.x**
* **PyTorch:** Core deep learning framework for tensor computation.
* **NLTK (Natural Language Toolkit):** For tokenization, stopword removal, and stemming.
* **Scikit-learn:** For TF-IDF vectorization and evaluation metrics.
* **Gensim:** For training and utilizing Word2Vec embeddings.
* **Pandas & NumPy:** For data manipulation and numerical operations.
* **Matplotlib:** For visualization of training loss and embedding clusters (PCA).
* **Google Colab (Recommended):** For access to GPU acceleration (CUDA).

## Implementation Details

### Part 1: Text Preprocessing and Baseline Models
This module establishes the foundational pipeline for cleaning raw text data and establishing a classification baseline using the IMDB Dataset.

* **Data Cleaning:** Implementation of regex-based cleaning to remove HTML tags, URLs, and special characters.
* **Normalization:** Application of lowercasing, stopword removal, and Stemming (PorterStemmer) to standardize tokens.
* **Vectorization:** Transformation of text into numerical vectors using Term Frequency-Inverse Document Frequency (TF-IDF).
* **Baseline Model:** Training a Logistic Regression classifier to evaluate initial performance metrics.
* **Word Embeddings (Advanced):** Implementation of Word2Vec using Gensim to generate dense vector representations and visualization using Principal Component Analysis (PCA).

### Part 2: Neural Network Architectures
This module transitions to deep learning approaches to capture complex patterns in text data.

* **Data Preparation:** Implementation of `CountVectorizer` (Bag of Words) and PyTorch `DataLoader` for batch processing.
* **Feedforward Network (ANN):**
    * **Architecture:** A fully connected network with a hidden layer utilizing ReLU activation and an output layer utilizing the Sigmoid function for binary classification.
    * **Training:** Optimization using the Adam optimizer and Binary Cross-Entropy Loss (BCELoss).
* **Recurrent Neural Network (RNN):**
    * **Architecture:** A network designed for sequential data, featuring an Embedding layer, an RNN layer to process hidden states, and a fully connected output layer.
    * **Objective:** To address the limitations of Bag of Words models by preserving the sequential order of words.


## Metrics

The models are evaluated based on the following key performance indicators:

* **Accuracy:** The overall correctness of the model in predicting sentiment (Positive/Negative).
* **Confusion Matrix:** A visualization of True Positives, True Negatives, False Positives, and False Negatives.
* **Loss Curves:** Visual tracking of training loss over epochs to detect convergence or overfitting.

## Acknowledgments

This project was developed as part of a comprehensive curriculum covering Natural Language Processing, NLTK, PyTorch, Recurrent Neural Networks (RNNs), LSTMs, and Transfer Learning with Hugging Face - [developed by Analytics Club, IIT Bombay](https://www.notion.so/Natural-Language-Processing-NLP-28f5dd880a0d805d82d4f1a396487ff4)
