from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

from retriever_faiss import FaissReviewRetriever
from generate_response import generate_contextual_response


# ------------------------------------------------------------
# 1. Basic configuration
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE_MODEL_NAME = "distilbert-base-uncased"
LORA_MODEL_DIR = PROJECT_ROOT / "models" / "sentiment_classifier_lora"

ID2LABEL = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


# ------------------------------------------------------------
# 2. Load classifier
# ------------------------------------------------------------

def load_lora_classifier():
    """
    Load the base DistilBERT classifier and attach the trained LoRA adapter.
    """

    tokenizer = AutoTokenizer.from_pretrained(str(LORA_MODEL_DIR))

    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id={
            "negative": 0,
            "neutral": 1,
            "positive": 2,
        },
    )

    model = PeftModel.from_pretrained(
        base_model,
        str(LORA_MODEL_DIR),
    )

    model.eval()

    return tokenizer, model


# ------------------------------------------------------------
# 3. Predict sentiment
# ------------------------------------------------------------

def predict_sentiment(text, tokenizer, model):
    """
    Predict negative / neutral / positive sentiment for a single text.
    """

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits

    predicted_class_id = torch.argmax(logits, dim=-1).item()

    predicted_label = ID2LABEL[predicted_class_id]

    return predicted_label


# ------------------------------------------------------------
# 4. Full pipeline test
# ------------------------------------------------------------

if __name__ == "__main__":

    user_query = "The food arrived cold and the delivery took forever. Nobody answered when I called."

    print("Loading classifier...")
    tokenizer, model = load_lora_classifier()

    print("Loading retriever...")
    retriever = FaissReviewRetriever()

    print("Predicting sentiment...")
    predicted_sentiment = predict_sentiment(user_query, tokenizer, model)

    print("Retrieving similar reviews...")
    retrieved_reviews = retriever.retrieve(user_query, top_k=3)

    print("Generating contextual response...")
    result = generate_contextual_response(
        user_query=user_query,
        predicted_sentiment=predicted_sentiment,
        retrieved_reviews=retrieved_reviews,
    )

    print()
    print("=" * 80)
    print("USER QUERY")
    print("=" * 80)
    print(user_query)

    print()
    print("=" * 80)
    print("PREDICTED SENTIMENT")
    print("=" * 80)
    print(predicted_sentiment)

    print()
    print("=" * 80)
    print("RETRIEVED CONTEXT")
    print("=" * 80)
    print(result["context_snippets"])

    print()
    print("=" * 80)
    print("GENERATED RESPONSE")
    print("=" * 80)
    print(result["response"])