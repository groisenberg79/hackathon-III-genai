import pickle
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer


# ------------------------------------------------------------
# 1. Basic configuration
# ------------------------------------------------------------

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

NUM_INDEX_EXAMPLES = 5000

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INDEX_DIR = PROJECT_ROOT / "models" / "faiss_index"
INDEX_PATH = INDEX_DIR / "reviews.index"
METADATA_PATH = INDEX_DIR / "review_metadata.pkl"


# ------------------------------------------------------------
# 2. Label mapping
# ------------------------------------------------------------

def map_yelp_label(label):
    """
    Yelp Review Full labels are originally:
        0 = 1 star
        1 = 2 stars
        2 = 3 stars
        3 = 4 stars
        4 = 5 stars

    We convert them into:
        negative
        neutral
        positive
    """

    if label in [0, 1]:
        return "negative"
    elif label == 2:
        return "neutral"
    else:
        return "positive"


# ------------------------------------------------------------
# 3. Main indexing pipeline
# ------------------------------------------------------------

if __name__ == "__main__":

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading Yelp dataset...")

    dataset = load_dataset("yelp_review_full")

    print("Selecting review subset...")

    reviews_dataset = (
        dataset["train"]
        .shuffle(seed=42)
        .select(range(NUM_INDEX_EXAMPLES))
    )

    print("Converting dataset to pandas DataFrame...")

    reviews_df = pd.DataFrame(reviews_dataset)

    print("Mapping labels to sentiment names...")

    reviews_df["sentiment"] = reviews_df["label"].apply(map_yelp_label)

    print("Loading embedding model...")

    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("Creating embeddings...")

    review_texts = reviews_df["text"].tolist()

    embeddings = embedding_model.encode(
        review_texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    print("Converting embeddings to float32...")

    embeddings = embeddings.astype("float32")

    embedding_dimension = embeddings.shape[1]

    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding dimension: {embedding_dimension}")

    print("Building FAISS index...")

    index = faiss.IndexFlatIP(embedding_dimension)

    index.add(embeddings)

    print(f"Number of vectors in FAISS index: {index.ntotal}")

    print("Preparing metadata...")

    metadata = reviews_df[["text", "label", "sentiment"]].to_dict(orient="records")

    print("Saving FAISS index...")

    faiss.write_index(index, str(INDEX_PATH))

    print("Saving metadata...")

    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)

    print(f"Done. FAISS index saved to: {INDEX_PATH}")
    print(f"Metadata saved to: {METADATA_PATH}")