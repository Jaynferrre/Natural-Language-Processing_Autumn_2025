# Natural-Language-Processing_Autumn_2025

## Project Overview

This repository contains the practical assignments, implementations, and the final capstone project associated with the Natural Language Processing (NLP) Bootcamp. The primary objective of this project is to explore and implement computational linguistics techniques and deep learning architectures for text classification tasks using Python, PyTorch, and the Hugging Face ecosystem.

The coursework progresses from fundamental text processing and traditional machine learning models to advanced neural network architectures and Transformer-based Transfer Learning. It provides a comparative analysis of their performance on sentiment analysis and multi-label emotion detection tasks.

## Repository Contents

The project is structured into three main components:

1.  **Assignment 1: Text Preprocessing and Baseline Classification**
    * Focuses on the fundamentals of Natural Language Understanding (NLU) using the Natural Language Toolkit (NLTK).
    * Implements traditional machine learning algorithms (Logistic Regression) as a performance baseline.
    * Explores semantic representation using Word Embeddings (Word2Vec).
2.  **Assignment 2: Deep Learning for Text Classification**
    * Focuses on building neural network architectures using PyTorch.
    * Implements Feedforward Neural Networks (Artificial Neural Networks) for text.
    * Implements Recurrent Neural Networks (RNN) to handle sequential data and capture temporal dependencies.
3.  **Capstone Project: Multilingual Emotion Detection**
    * Focuses on Transfer Learning using Transformer architectures (DistilBERT & mBERT).
    * Addresses Multi-label Classification on the SemEval-2018 Task 1 dataset.
    * Evaluates Zero-Shot Cross-Lingual transfer capabilities.

## Technical Prerequisites

To execute the code contained in this repository, the following software and libraries are required:

* **Python 3.x**
* **PyTorch:** Core deep learning framework for tensor computation.
* **Hugging Face Transformers:** For pre-trained BERT models and Trainer API.
* **NLTK (Natural Language Toolkit):** For tokenization, stopword removal, and stemming.
* **Scikit-learn:** For TF-IDF vectorization and evaluation metrics.
* **Gensim:** For training and utilizing Word2Vec embeddings.
* **Pandas & NumPy:** For data manipulation and numerical operations.
* **Matplotlib & Seaborn:** For visualization of training loss, embedding clusters (PCA), and confusion matrices.
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

### Part 3: Capstone Project (Multilingual Emotion Detection)
This module applies Transfer Learning to solve a complex Multi-label Classification problem using the **SemEval-2018 Task 1 (Affect in Tweets)** dataset.

* **Objective:** To detect four distinct emotions (Anger, Joy, Love, Pessimism) in tweets and evaluate the ability of models to generalize across languages.
* **Models Implemented:**
    * **Monolingual:** Fine-tuned `DistilBERT-base-uncased` for efficient, high-accuracy English classification.
    * **Multilingual:** Fine-tuned `BERT-base-multilingual-cased` (mBERT) to test zero-shot transfer capabilities on non-English text.
* **Methodology:**
    * Utilized the Hugging Face `Trainer` API for fine-tuning.
    * Implemented `BCEWithLogitsLoss` for multi-label handling.
    * Compared performance metrics (F1-Score) to analyze the trade-off between specialized (monolingual) and generalized (multilingual) architectures.
* **Key Findings:**
    * DistilBERT outperformed mBERT in distinct emotions like "Anger" (+1.4%) and "Joy" (+3%).
    * mBERT demonstrated superior performance in nuanced categories like "Love" (+3.7%) and successfully classified multilingual queries (Zero-Shot Transfer).

## Metrics

The models are evaluated based on the following key performance indicators:

* **Accuracy & F1-Score:** The overall correctness and harmonic mean of precision and recall.
* **Confusion Matrix:** A visualization of True Positives, True Negatives, False Positives, and False Negatives (Critical for the multi-label Capstone analysis).
* **Loss Curves:** Visual tracking of training loss over epochs to detect convergence or overfitting.

## Acknowledgments

This project was developed as part of a comprehensive curriculum covering Natural Language Processing, NLTK, PyTorch, Recurrent Neural Networks (RNNs), LSTMs, and Transfer Learning with Hugging Face - [developed by Analytics Club, IIT Bombay](https://www.notion.so/Natural-Language-Processing-NLP-28f5dd880a0d805d82d4f1a396487ff4)
