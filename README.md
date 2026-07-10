# Multilingual Intent Classifier using LaBSE

A multilingual intent classification system built using **LaBSE (Language-Agnostic BERT Sentence Embedding)** and a **Logistic Regression** classifier. The project classifies user queries into predefined intents without using a Large Language Model (LLM) for inference.

## Overview

This project follows an embedding-based intent classification pipeline:

1. Dataset Construction
2. Sentence Embedding Generation using LaBSE
3. Embedding Caching
4. Classifier Training
5. Model Evaluation
6. Intent Prediction

The system supports multilingual queries in:

- English
- Hindi
- Hinglish

---

## Intent Categories

The classifier predicts one of the following intents:

- Conversation
- Search
- Calendar
- Reminder
- Task
- Contact
- Camera
- Device

---

## Dataset

- Approximately 800 manually curated examples
- Queries in English, Hindi and Hinglish
- Duplicate removal and dataset validation
- Eight predefined intent classes

Dataset format:

| sentence | intent |
|----------|--------|
| Hello | conversation |
| Call Rahul | contact |
| Take a picture | camera |

---

## Project Structure

```text
GIN_Intent_Classifier/
│
├── dataset/
│   ├── intent_dataset.csv
│   └── intent_dataset_clean.csv
│
├── embeddings/
│   └── labse_embeddings.npy
│
├── models/
│   ├── logistic_regression.pkl
│   └── label_encoder.pkl
│
├── generate_embeddings.py
├── train_classifier.py
├── evaluate.py
├── predict.py
├── requirements.txt
└── README.md
```

---

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Sentence Transformers
- Joblib

### Embedding Model

- LaBSE (`sentence-transformers/LaBSE`)

### Classifier

- Logistic Regression

---

## Pipeline

### 1. Dataset Construction

- Curated multilingual dataset
- Assigned intent labels
- Removed duplicate samples

### 2. Embedding Generation

Generated sentence embeddings using **LaBSE**.

Each sentence is converted into a **768-dimensional dense vector** capturing its semantic meaning across multiple languages.

Embeddings are cached to avoid recomputation during future experiments.

### 3. Classifier Training

- Label Encoding
- Train-Test Split (80:20)
- Logistic Regression

### 4. Evaluation

Performance was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

### 5. Prediction

Given a user query, the system:

1. Generates a LaBSE embedding.
2. Passes the embedding to the trained classifier.
3. Predicts the corresponding intent.

Example:

Input:

```
Call Rahul
```

Output:

```
Predicted Intent: contact
```

---

## Results

| Metric | Value |
|---------|------:|
| Accuracy | **93.13%** |
| Macro F1-score | **0.93** |

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/<repository-name>.git
```

Navigate to the project directory:

```bash
cd <repository-name>
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Generate Embeddings

```bash
python generate_embeddings.py
```

### Train the Classifier

```bash
python train_classifier.py
```

### Evaluate the Model

```bash
python evaluate.py
```

### Predict Intent

```bash
python predict.py
```

---

## Future Work

- Evaluate additional multilingual embedding models
- Compare different machine learning classifiers
- Expand the multilingual dataset
- Improve performance on ambiguous intent classes
- Visualize the confusion matrix

---

## Author

Noor Tandon
Computer Engineering
