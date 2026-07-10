import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer



EMBEDDING_MODEL = "sentence-transformers/LaBSE"



df = pd.read_csv(
    "dataset/intent_dataset_clean.csv",
    encoding="utf-8" #taaki hindi sentences bhi visible ho
)

sentences = df["sentence"].tolist()

print("=" * 60)
print("Embedding Generation Started")
print("=" * 60)

print(f"\nEmbedding Model : {EMBEDDING_MODEL}")

print(f"\nTotal Sentences : {len(sentences)}")



print("\nLoading embedding model...")

model = SentenceTransformer(EMBEDDING_MODEL)

print("Model loaded successfully!")



print("\nGenerating embeddings...")

embeddings = model.encode(
    sentences,
    show_progress_bar=True,
    convert_to_numpy=True
)

print("\nEmbedding Shape :", embeddings.shape)


np.save(
    "labse_embeddings.npy",
    embeddings
)

print("\nEmbeddings saved successfully!")