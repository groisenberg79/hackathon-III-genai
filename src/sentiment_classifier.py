from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel


# ------------------------------------------------------------
# 1. Basic configuration
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

BASE_MODEL_NAME = "distilbert-base-uncased"
LORA_MODEL_DIR = PROJECT_ROOT / "models" / "sentiment_classifier_lora_random_10k"

MAX_LENGTH = 128

ID2LABEL = {
    0: "negative",
    1: "neutral",
    2: "positive",
}

LABEL2ID = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}


# ------------------------------------------------------------
# 2. Device helper
# ------------------------------------------------------------

def get_device():
    """
    Choose the best available device for the sentiment classifier.

    Priority:
        1. Apple Silicon MPS
        2. NVIDIA CUDA
        3. CPU
    """

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ------------------------------------------------------------
# 3. Validation helper
# ------------------------------------------------------------

def validate_model_files():
    """
    Check that the trained LoRA model folder exists.

    The models/ folder is usually excluded from Git, so a fresh clone of the
    repository will need to recreate this folder by running the training script.
    """

    if not LORA_MODEL_DIR.exists():
        raise FileNotFoundError(
            f"LoRA model directory not found:\n"
            f"{LORA_MODEL_DIR}\n\n"
            f"Run this first from the project root:\n"
            f"python src/train_classifier.py"
        )


# ------------------------------------------------------------
# 4. Load LoRA sentiment classifier
# ------------------------------------------------------------

def load_lora_classifier():
    """
    Load the trained LoRA sentiment classifier.

    The LoRA adapter is saved separately, so we load:
        1. the base DistilBERT sequence-classification model
        2. the LoRA adapter from models/sentiment_classifier_lora
    """

    validate_model_files()

    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(str(LORA_MODEL_DIR))

    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    model = PeftModel.from_pretrained(
        base_model,
        str(LORA_MODEL_DIR),
    )

    model.to(device)
    model.eval()

    return tokenizer, model, device


# ------------------------------------------------------------
# 5. Predict sentiment
# ------------------------------------------------------------

def predict_sentiment(text, tokenizer, model, device):
    """
    Predict negative / neutral / positive sentiment for one text.

    Returns:
        predicted_label: str
    """

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=-1).item()

    predicted_label = ID2LABEL[predicted_class_id]

    return predicted_label


# ------------------------------------------------------------
# 6. Predict sentiment with scores
# ------------------------------------------------------------

def predict_sentiment_with_scores(text, tokenizer, model, device):
    """
    Predict sentiment and return probabilities for all classes.

    This is useful for debugging and for the Streamlit UI.
    """

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits

    probabilities = torch.softmax(logits, dim=-1)[0].detach().cpu()

    scores = {
        ID2LABEL[class_id]: float(probabilities[class_id])
        for class_id in ID2LABEL
    }

    predicted_class_id = int(torch.argmax(probabilities).item())
    predicted_label = ID2LABEL[predicted_class_id]

    return {
        "label": predicted_label,
        "scores": scores,
    }


# ------------------------------------------------------------
# 7. Manual test
# ------------------------------------------------------------

if __name__ == "__main__":

    tokenizer, model, device = load_lora_classifier()

    print(f"Loaded classifier on device: {device}")

    test_texts = [
        "The food arrived cold and the delivery took forever.",
        "The meal was fine, nothing amazing, but the staff were polite.",
        "The food was delicious, the service was friendly, and everything arrived quickly.",
    ]

    for text in test_texts:
        result = predict_sentiment_with_scores(
            text=text,
            tokenizer=tokenizer,
            model=model,
            device=device,
        )

        print("=" * 80)
        print(text)
        print(f"Predicted sentiment: {result['label']}")
        print("Scores:")

        for label, score in result["scores"].items():
            print(f"  {label}: {score:.4f}")