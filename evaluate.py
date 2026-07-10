import numpy as np
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# =====================================
# Load Embeddings
# =====================================

X = np.load("embeddings\labse_embeddings.npy")

# =====================================
# Load Dataset
# =====================================

df = pd.read_csv("dataset/intent_dataset_clean.csv")

# =====================================
# Encode Labels
# =====================================

encoder = LabelEncoder()

y = encoder.fit_transform(df["intent"])

# =====================================
# Train-Test Split
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =====================================
# Load Trained Model
# =====================================

classifier = joblib.load(
    "models/logistic_regression.pkl"
)

# =====================================
# Predict
# =====================================

y_pred = classifier.predict(X_test)

# =====================================
# Accuracy
# =====================================

accuracy = accuracy_score(y_test, y_pred)

print("=" * 50)
print(f"Accuracy : {accuracy:.4f}")
print("=" * 50)

# =====================================
# Classification Report
# =====================================

print("\nClassification Report\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=encoder.classes_
    )
)

# =====================================
# Confusion Matrix
# =====================================

print("\nConfusion Matrix\n")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)