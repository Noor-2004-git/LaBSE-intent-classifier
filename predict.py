import joblib
from sentence_transformers import SentenceTransformer

# ==========================================
# Configuration
# ==========================================

EMBEDDING_MODEL = "sentence-transformers/LaBSE"

# ==========================================
# Load Embedding Model
# ==========================================

print("Loading embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# ==========================================
# Load Classifier
# ==========================================

print("Loading classifier...")

classifier = joblib.load("models/logistic_regression.pkl")
label_encoder = joblib.load("models/label_encoder.pkl")

print("Model loaded successfully!\n")

# ==========================================
# Prediction Loop
# ==========================================

while True:

    query = input("Enter your query (type 'exit' to quit): ")

    if query.lower() == "exit":
        print("Exiting...")
        break

    # Generate embedding
    embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    )

    # Predict intent
    prediction = classifier.predict(embedding)

    intent = label_encoder.inverse_transform(prediction)[0]

    print(f"\nPredicted Intent : {intent}\n")