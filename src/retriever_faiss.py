import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# ------------------------------------------------------------
# 1. Basic configuration
# ------------------------------------------------------------

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INDEX_DIR = PROJECT_ROOT / "models" / "faiss_index"
INDEX_PATH = INDEX_DIR / "reviews.index"
METADATA_PATH = INDEX_DIR / "review_metadata.pkl"


# ------------------------------------------------------------
# 2. FAISS retriever class
# ------------------------------------------------------------

class FaissReviewRetriever:
    """
    Loads a saved FAISS index and retrieves reviews that are semantically
    similar to a user query.
    """

    def __init__(self):
        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

        print("Loading FAISS index...")
        self.index = faiss.read_index(str(INDEX_PATH))

        print("Loading metadata...")
        with open(METADATA_PATH, "rb") as f:
            self.metadata = pickle.load(f)

    def retrieve(self, query, top_k=3):
        """
        Retrieve the top_k most similar reviews for a user query.

        Parameters:
            query:
                The user's input text.

            top_k:
                Number of similar reviews to retrieve.

        Returns:
            A list of dictionaries containing:
                - score
                - text
                - original label
                - sentiment
        """

        query_embedding = self.embedding_model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        query_embedding = query_embedding.astype("float32")

        scores, indices = self.index.search(query_embedding, top_k)

        results = []

        for score, index_position in zip(scores[0], indices[0]):
            item = self.metadata[index_position]

            results.append(
                {
                    "score": float(score),
                    "text": item["text"],
                    "label": item["label"],
                    "sentiment": item["sentiment"],
                }
            )

        return results


# ------------------------------------------------------------
# 3. Manual test
# ------------------------------------------------------------

if __name__ == "__main__":

    retriever = FaissReviewRetriever()

    test_query = "The food arrived cold and the delivery was very late."

    print("=" * 80)
    print("Query:")
    print(test_query)
    print("=" * 80)

    results = retriever.retrieve(test_query, top_k=3)

    for i, result in enumerate(results, start=1):
        print()
        print(f"Result {i}")
        print("-" * 80)
        print(f"Score: {result['score']:.4f}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Original Yelp label: {result['label']}")
        print(f"Review text: {result['text'][:700]}")