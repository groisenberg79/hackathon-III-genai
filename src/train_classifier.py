import numpy as np
from pathlib import Path

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# ------------------------------------------------------------
# 1. Basic configuration
# ------------------------------------------------------------

MODEL_NAME = "distilbert-base-uncased"

# Random sample sizes.
# This is the same general approach as the earlier successful LoRA run,
# but with more examples.
NUM_TRAIN_EXAMPLES = 10000
NUM_VALIDATION_EXAMPLES = 1500

MAX_LENGTH = 128

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# IMPORTANT:
# Save this experiment to a new folder so we do not overwrite the previous model.
OUTPUT_DIR = PROJECT_ROOT / "models" / "sentiment_classifier_lora_random_10k"


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

    Mapping:
        1–2 stars -> negative
        3 stars   -> neutral
        4–5 stars -> positive
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
    Convert raw review text into model inputs.

    The tokenizer returns:
        input_ids:
            Numerical token IDs.

        attention_mask:
            1 for real tokens, 0 for padding tokens.

    We use:
        truncation=True:
            Cut reviews longer than MAX_LENGTH.

        padding="max_length":
            Pad shorter reviews to MAX_LENGTH.

        max_length=128:
            Keeps training reasonably fast for the hackathon.
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
    Compute evaluation metrics for the validation set.

    Trainer gives us:
        logits:
            Raw model scores before softmax.

        labels:
            Correct class labels.

    We convert logits into predicted classes using argmax.
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

    print("=" * 80)
    print("Loading dataset...")
    print("=" * 80)

    dataset = load_dataset("yelp_review_full")

    print()
    print("=" * 80)
    print("Creating random train/validation subsets...")
    print("=" * 80)

    train_dataset = (
        dataset["train"]
        .shuffle(seed=42)
        .select(range(NUM_TRAIN_EXAMPLES))
    )

    validation_dataset = (
        dataset["test"]
        .shuffle(seed=123)
        .select(range(NUM_VALIDATION_EXAMPLES))
    )

    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(validation_dataset)}")

    print()
    print("=" * 80)
    print("Mapping 5-star Yelp labels into 3 sentiment labels...")
    print("=" * 80)

    train_dataset = train_dataset.map(map_yelp_label)
    validation_dataset = validation_dataset.map(map_yelp_label)

    print()
    print("=" * 80)
    print("Loading tokenizer...")
    print("=" * 80)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print()
    print("=" * 80)
    print("Tokenizing datasets...")
    print("=" * 80)

    train_dataset = train_dataset.map(
        tokenize_examples,
        batched=True,
    )

    validation_dataset = validation_dataset.map(
        tokenize_examples,
        batched=True,
    )

    train_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"],
    )

    validation_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "label"],
    )

    print()
    print("=" * 80)
    print("Loading base model...")
    print("=" * 80)

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

    print()
    print("=" * 80)
    print("Configuring LoRA...")
    print("=" * 80)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,

        # LoRA rank. Larger rank = more trainable parameters.
        # r=8 is a reasonable small/medium value for this hackathon project.
        r=8,

        # Common rule of thumb: alpha around 2 * r.
        lora_alpha=16,

        # Dropout inside LoRA adapters for regularization.
        lora_dropout=0.1,

        # DistilBERT attention projection layer names.
        # q_lin = query projection
        # v_lin = value projection
        target_modules=["q_lin", "v_lin"],

        # Keep classification head trainable/saved.
        modules_to_save=["classifier", "pre_classifier"],
    )

    model = get_peft_model(model, lora_config)

    print("Trainable parameters after applying LoRA:")
    model.print_trainable_parameters()

    print()
    print("=" * 80)
    print("Setting up training arguments...")
    print("=" * 80)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),

        # Evaluate and save after each epoch.
        eval_strategy="epoch",
        save_strategy="epoch",

        # LoRA often works well with a slightly larger LR than full fine-tuning.
        learning_rate=1e-4,

        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,

        # Keep this at 2 for hackathon speed.
        # If you have time, 3 epochs may improve the classifier.
        num_train_epochs=2,

        weight_decay=0.01,
        logging_steps=50,

        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,

        # Avoid W&B or other tracking prompts.
        report_to="none",

        # Avoid MPS pinned-memory warning on Apple Silicon.
        dataloader_pin_memory=False,
    )

    print()
    print("=" * 80)
    print("Creating Trainer...")
    print("=" * 80)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=compute_metrics,
    )

    print()
    print("=" * 80)
    print("Training random 10k LoRA sentiment classifier...")
    print("=" * 80)

    trainer.train()

    print()
    print("=" * 80)
    print("Evaluating LoRA model...")
    print("=" * 80)

    results = trainer.evaluate()
    print(results)

    print()
    print("=" * 80)
    print("Saving LoRA model and tokenizer...")
    print("=" * 80)

    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    print()
    print("=" * 80)
    print("Done.")
    print("=" * 80)
    print(f"LoRA model saved to: {OUTPUT_DIR}")