import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

# ===========================================
# Load Embeddings
# ===========================================
print("Script Started")


X = np.load("embeddings/labse_embeddings.npy")

print("Embeddings Loaded")
print(X.shape)

# ===========================================
# Load Dataset
# ===========================================

df = pd.read_csv("dataset/intent_dataset_clean.csv")

# ===========================================
# Encode Labels
# ===========================================

encoder = LabelEncoder()

y = encoder.fit_transform(df["intent"])

print("\nIntent Mapping")

for i, label in enumerate(encoder.classes_):
    print(f"{i} -> {label}")

# ===========================================
# Train Test Split
# ===========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples :", len(X_test))

# ===========================================
# Train Logistic Regression
# ===========================================

classifier = LogisticRegression(
    max_iter=1000
)

classifier.fit(X_train, y_train)

print("\nTraining Completed Successfully!")

# ===========================================
# Save Model
# ===========================================

joblib.dump(
    classifier,
    "models/logistic_regression.pkl"
)

joblib.dump(
    encoder,
    "models/label_encoder.pkl"
)

print("\nModel Saved!")