📊 Text Classification: Machine Learning vs Neural Networks

A Comparative Analysis on a Question–Answer Dataset

🔍 Project Overview

This project presents a comparative study of classical Machine Learning (ML) models and Neural Network (NN) architectures for multi-class text classification.
The task focuses on categorizing Question–Answer (QA) text data into predefined classes using different feature representations and model architectures.

The goal was to understand:

How traditional ML models compare against deep learning approaches

The impact of different text representation techniques

Which model–feature combination performs best for QA-style text data

📌 Best-performing model:
Bidirectional GRU (BiGRU) with pre-trained GloVe embeddings, achieving a weighted F1-score of 63.02%.

🧠 Key Contributions

Implemented 10 different ML and NN models

Compared BoW, TF-IDF, Word2Vec (Skip-gram), and GloVe embeddings

Performed systematic hyperparameter tuning

Evaluated models using Accuracy and Weighted F1-score

Analyzed class-wise performance using confusion matrices

📁 Dataset

Type: Multi-class Question–Answer text dataset

Structure:

Question Title

Question Content

Best Answer

Class Label

Split: Predefined training and testing sets

Nature: Public dataset provided as part of university coursework

⚠️ Due to GitHub file size limitations, the dataset is not included directly in this repository.
You can plug in the dataset locally by updating the file paths in the notebook.

🧹 Data Preprocessing

A complete NLP preprocessing pipeline was applied:

Lowercasing and text normalization

Removal of non-alphabetic characters

Tokenization

Stopword removal (NLTK)

Lemmatization (WordNet)

Stemming

This ensured a clean, normalized corpus for feature extraction.

🧩 Feature Representation Techniques

The following text representations were evaluated:

Representation	Description
Bag-of-Words (BoW)	Frequency-based representation (top 5,000 terms)
TF-IDF	Weighted representation emphasizing discriminative words
Word2Vec (Skip-gram)	Custom-trained 100D embeddings
GloVe	Pre-trained 100D embeddings capturing global semantics
🤖 Models Implemented
🔹 Machine Learning Models

Logistic Regression

Multinomial Naive Bayes

Random Forest

(trained using BoW and TF-IDF)

🔹 Neural Network Models

Deep Feedforward Neural Network

Simple RNN

Bidirectional RNN

LSTM

Bidirectional LSTM

GRU

Bidirectional GRU

(trained using TF-IDF, Word2Vec, and GloVe embeddings)

Neural models included:

Dropout regularization

Early stopping

Fixed embedding layers (non-trainable)

⚙️ Hyperparameter Tuning

Batch size: 32

Epochs: 5 (to prevent overfitting)

Dropout: 0.5

Neurons per dense layer: 128

Feature extraction parameters (n-grams, vocabulary size) were tuned for ML models.

📈 Results Summary
🏆 Top Performing Models
Model	Accuracy (%)	F1-score (%)
BiGRU (GloVe)	62.66	63.02
GRU (GloVe)	61.02	61.54
LSTM (GloVe)	61.06	61.50
Random Forest (TF-IDF)	—	62.04
Logistic Regression (TF-IDF)	56.79	57.54
📉 Observations

GloVe embeddings consistently outperformed Skip-gram

TF-IDF > BoW for classical ML models

GRU/LSTM architectures handled long-range dependencies better

Simple RNNs performed poorly, failing to capture contextual information

Random Forest (TF-IDF) proved to be a strong and efficient baseline

📊 Visual Analysis

F1-score and accuracy comparisons across all models

Confusion matrices for best-performing models

Clear separation into top, middle, and bottom performance tiers

🧠 Key Takeaways

Pre-trained embeddings + bidirectional architectures are highly effective

Classical ML models remain competitive when paired with strong features

Model choice should balance performance vs computational cost

🚀 Future Work

Experiment with Transformer-based models (BERT, RoBERTa)

Apply automated hyperparameter search

Fine-tune pre-trained embeddings on domain-specific data

Explore advanced regularization techniques

🛠 Tech Stack

Python

NumPy, Pandas

Scikit-learn

TensorFlow / Keras

NLTK

Gensim

Matplotlib, Seaborn

📄 Report

A detailed academic report describing methodology, experiments, and analysis is available in the report/ directory.

👤 Authors

Dibya Ayishwarja Mallick
Computer Science Undergraduate, BRAC University
Dhaka, Bangladesh

Fayruz Tahania Haseen
Computer Science Undergraduate, BRAC University
Dhaka, Bangladesh

Md Faiyaz
Computer Science Undergraduate, BRAC University
Dhaka, Bangladesh