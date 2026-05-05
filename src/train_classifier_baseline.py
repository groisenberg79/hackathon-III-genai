from pathlib import Path
import numpy as np

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# ------------------------------------------------------------
# 1. Basic configuration
# ------------------------------------------------------------

MODEL_NAME = "distilbert-base-uncased"

NUM_TRAIN_EXAMPLES = 6000
NUM_VALIDATION_EXAMPLES = 1000

MAX_LENGTH = 128

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "models" / "sentiment_classifier"


# ------------------------------------------------------------
# 2. Convert Yelp's 5-star labels into 3 sentiment classes
# ------------------------------------------------------------

def map_yelp_label(example):
    """
    Yelp Review Full labels are originally:
        0 = 1 star
        1 = 2 stars
        2 = 3 stars
        3 = 4 stars
        4 = 5 stars

    We convert them into:
        0 = negative
        1 = neutral
        2 = positive
    """

    original_label = example["label"]

    if original_label in [0, 1]:
        example["label"] = 0
    elif original_label == 2:
        example["label"] = 1
    else:
        example["label"] = 2

    return example


# ------------------------------------------------------------
# 3. Tokenization function
# ------------------------------------------------------------

def tokenize_examples(examples):
    """
    The tokenizer converts raw review text into input IDs and attention masks.

    input_ids:
        Numerical token IDs that the model can understand.

    attention_mask:
        Tells the model which tokens are real and which are padding.
    """

    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
    )


# ------------------------------------------------------------
# 4. Evaluation metrics
# ------------------------------------------------------------

def compute_metrics(eval_pred):
    """
    Trainer gives us:
        logits: raw model scores before softmax
        labels: true labels

    We choose the class with the highest logit as the prediction.
    """

    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ------------------------------------------------------------
# 5. Main training pipeline
# ------------------------------------------------------------

if __name__ == "__main__":

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")

    dataset = load_dataset("yelp_review_full")

    train_dataset = dataset["train"].shuffle(seed=42).select(range(NUM_TRAIN_EXAMPLES))
    validation_dataset = dataset["test"].shuffle(seed=42).select(range(NUM_VALIDATION_EXAMPLES))

    print("Mapping 5-star Yelp labels into 3 sentiment labels...")

    train_dataset = train_dataset.map(map_yelp_label)
    validation_dataset = validation_dataset.map(map_yelp_label)

    print("Loading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Tokenizing datasets...")

    train_dataset = train_dataset.map(tokenize_examples, batched=True)
    validation_dataset = validation_dataset.map(tokenize_examples, batched=True)

    train_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"],
    )

    validation_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"],
    )

    print("Loading model...")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label={
            0: "negative",
            1: "neutral",
            2: "positive",
        },
        label2id={
            "negative": 0,
            "neutral": 1,
            "positive": 2,
        },
    )

    print("Setting up training arguments...")

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=2,
        weight_decay=0.01,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        report_to="none",
    )

    print("Creating Trainer...")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=compute_metrics,
    )

    print("Training model...")

    trainer.train()

    print("Evaluating model...")

    results = trainer.evaluate()
    print(results)

    print("Saving model and tokenizer...")

    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    print(f"Done. Model saved to {OUTPUT_DIR}")