from sentiment_classifier import load_lora_classifier, predict_sentiment
from retriever_faiss import FaissReviewRetriever
from generate_response_remote import generate_contextual_response


# ------------------------------------------------------------
# 1. Print retrieved reviews
# ------------------------------------------------------------

def print_retrieved_reviews(retrieved_reviews):
    """
    Print retrieved reviews in a readable way.
    """

    for i, review in enumerate(retrieved_reviews, start=1):
        text = review["text"].replace("\n", " ").strip()

        if len(text) > 500:
            text = text[:500].rstrip() + "..."

        print()
        print(f"Retrieved Review {i}")
        print("-" * 80)
        print(f"Similarity score: {review['score']:.4f}")
        print(f"Sentiment: {review['sentiment']}")
        print(f"Original Yelp label: {review['label']}")
        print(f"Text: {text}")


# ------------------------------------------------------------
# 2. Load reusable pipeline components
# ------------------------------------------------------------

def load_pipeline_components():
    """
    Load reusable local components once.

    OpenRouter is called remotely during generation, so there is no local
    generator model to load.
    """

    print("Loading LoRA sentiment classifier...")
    classifier_tokenizer, classifier_model, classifier_device = load_lora_classifier()

    print()
    print("Loading FAISS retriever...")
    retriever = FaissReviewRetriever()

    return {
        "classifier_tokenizer": classifier_tokenizer,
        "classifier_model": classifier_model,
        "classifier_device": classifier_device,
        "retriever": retriever,
    }


# ------------------------------------------------------------
# 3. Run full pipeline for one query
# ------------------------------------------------------------

def run_pipeline(user_query, components, top_k=3):
    """
    Run the full pipeline:

        user query
            ↓
        LoRA sentiment classifier
            ↓
        predicted sentiment

        user query + predicted sentiment
            ↓
        FAISS retrieval with sentiment filtering
            ↓
        similar reviews

        user query + predicted sentiment + retrieved reviews
            ↓
        OpenRouter Llama generator
            ↓
        customer support response
    """

    print()
    print("Predicting sentiment...")

    predicted_sentiment = predict_sentiment(
        text=user_query,
        tokenizer=components["classifier_tokenizer"],
        model=components["classifier_model"],
        device=components["classifier_device"],
    )

    print("Retrieving similar reviews with sentiment filtering...")

    retrieved_reviews = components["retriever"].retrieve_with_sentiment_filter(
        query=user_query,
        sentiment=predicted_sentiment,
        top_k=top_k,
        candidate_k=20,
    )

    print("Generating contextual response with OpenRouter...")

    generation_result = generate_contextual_response(
        user_query=user_query,
        predicted_sentiment=predicted_sentiment,
        retrieved_reviews=retrieved_reviews,
    )

    return {
        "user_query": user_query,
        "predicted_sentiment": predicted_sentiment,
        "retrieved_reviews": retrieved_reviews,
        "generation_result": generation_result,
    }


# ------------------------------------------------------------
# 4. Manual test
# ------------------------------------------------------------

if __name__ == "__main__":

    test_queries = [
        "The food arrived cold and the delivery took forever. Nobody answered when I called.",
        "The meal was fine, nothing amazing, but the staff were polite.",
        "The food was delicious, the service was friendly, and everything arrived quickly.",
    ]

    components = load_pipeline_components()

    for query in test_queries:
        result = run_pipeline(
            user_query=query,
            components=components,
            top_k=3,
        )

        print()
        print("=" * 100)
        print("USER QUERY")
        print("=" * 100)
        print(result["user_query"])

        print()
        print("=" * 100)
        print("PREDICTED SENTIMENT")
        print("=" * 100)
        print(result["predicted_sentiment"])

        print()
        print("=" * 100)
        print("RETRIEVED REVIEWS")
        print("=" * 100)
        print_retrieved_reviews(result["retrieved_reviews"])

        print()
        print("=" * 100)
        print("GENERATOR MODEL")
        print("=" * 100)
        print(result["generation_result"]["model"])

        print()
        print("=" * 100)
        print("FINAL RESPONSE")
        print("=" * 100)
        print(result["generation_result"]["response"])

        print()
        print("#" * 100)
        print("#" * 100)
        print()