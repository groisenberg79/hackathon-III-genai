import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ------------------------------------------------------------
# 1. Basic configuration
# ------------------------------------------------------------

GENERATION_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


# ------------------------------------------------------------
# 2. Load generator
# ------------------------------------------------------------

def load_generator():
    """
    Load the tokenizer and local generative model.

    This model is small enough to run locally, but is still instruction-tuned,
    so it should be better suited for customer-support-style generation than
    FLAN-T5-small/base in this project.
    """

    tokenizer = AutoTokenizer.from_pretrained(GENERATION_MODEL_NAME)

    # `torch_dtype` works but now raises a deprecation warning in newer
    # transformers versions. `dtype` is the newer argument name.
    model = AutoModelForCausalLM.from_pretrained(
        GENERATION_MODEL_NAME,
        dtype=torch.float32,
    )

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model.to(device)
    model.eval()

    print(f"Generator loaded on device: {device}")

    return tokenizer, model, device


# ------------------------------------------------------------
# 3. Infer issue summary from user query + retrieved reviews
# ------------------------------------------------------------

def infer_issue_summary(user_query, retrieved_reviews):
    """
    Build a short issue summary from the user query and retrieved reviews.

    This keeps the generation prompt compact and avoids making the model copy
    full retrieved reviews.
    """

    combined_text = user_query.lower()

    for review in retrieved_reviews[:2]:
        combined_text += " " + review["text"].lower()

    issues = []

    if any(word in combined_text for word in ["cold", "temperature", "lukewarm"]):
        issues.append("food arrived cold")

    if any(word in combined_text for word in ["late", "delay", "delayed", "forever", "wait", "waiting"]):
        issues.append("delivery or service was delayed")

    if any(
        phrase in combined_text
        for phrase in ["nobody answered", "no one answered", "did not answer", "ignored"]
    ):
        issues.append("customer could not reach support")

    if any(word in combined_text for word in ["rude", "disrespectful", "unfriendly"]):
        issues.append("staff interaction was disappointing")

    if any(
        phrase in combined_text
        for phrase in ["wrong item", "missing item", "incorrect order"]
    ):
        issues.append("order accuracy problem")

    if not issues:
        issues.append("customer had a notable experience")

    return ", ".join(issues)


# ------------------------------------------------------------
# 4. Build retrieved context snippets for display
# ------------------------------------------------------------

def build_retrieved_context_for_display(retrieved_reviews, max_reviews=3, max_chars=220):
    """
    Prepare retrieved reviews for display in Streamlit.

    These snippets help show that retrieval is actually happening.
    """

    snippets = []

    for i, review in enumerate(retrieved_reviews[:max_reviews], start=1):
        text = review["text"].replace("\n", " ").strip()

        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "..."

        snippets.append(
            {
                "rank": i,
                "score": review["score"],
                "sentiment": review["sentiment"],
                "text": text,
            }
        )

    return snippets


# ------------------------------------------------------------
# 5. Build chat messages
# ------------------------------------------------------------

def build_generation_messages(user_query, predicted_sentiment, retrieved_reviews):
    """
    Build chat-style messages for the local instruction model.
    """

    issue_summary = infer_issue_summary(user_query, retrieved_reviews)

    system_message = """
        You are a professional customer support assistant for a restaurant or delivery business.
        You write replies from the business to the customer.
        Do not write a customer review.
        Do not repeat the customer complaint word-for-word.
        Do not mention AI, classifiers, embeddings, FAISS, retrieval, or similar reviews.
        Do not invent a refund policy.
        Do not add specific details that the customer did not provide.
        Do not use placeholders such as [Customer's Name], [Your Name], or [Company Name].
        Do not include greetings, signatures, or subject lines.
    """.strip()

    user_message = f"""
Customer message:
{user_query}

Predicted sentiment:
{predicted_sentiment}

Main issues:
{issue_summary}

Write only the final customer support reply.
Write exactly 3 short sentences.
Sentence 1: apologize or acknowledge the experience.
Sentence 2: briefly mention the main issue.
Sentence 3: ask for order or visit details so the team can follow up.
Do not include a greeting.
Do not include a sign-off.
""".strip()

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


# ------------------------------------------------------------
# 6. Clean generated response
# ------------------------------------------------------------

def clean_generated_response(text):
    """
    Clean common artifacts from model output.
    """

    text = text.strip()

    # Remove accidental leading labels.
    text = re.sub(
        r"^(assistant|customer support response|response)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Remove placeholder names/signatures that small chat models sometimes generate.
    text = re.sub(r"Dear\s+\[[^\]]+\],?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Best regards,?\s*\[[^\]]+\].*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[[^\]]+\]", "", text)

    # Collapse excessive whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ------------------------------------------------------------
# 7. Safe fallback response
# ------------------------------------------------------------

def build_safe_fallback_response(user_query, predicted_sentiment, retrieved_reviews):
    """
    Fallback used if the local generator produces unusable output.

    This protects the Streamlit demo, but the goal is still to use the
    generated response whenever it is reasonably acceptable.
    """

    issue_summary = infer_issue_summary(user_query, retrieved_reviews)

    if predicted_sentiment == "negative":
        return (
            f"I'm sorry to hear about this experience. "
            f"We understand how frustrating it can be when {issue_summary}. "
            f"Please share your order or visit details so our team can look into what happened and follow up with an appropriate next step."
        )

    if predicted_sentiment == "positive":
        return (
            "Thank you for sharing your feedback. "
            "We're glad to hear that you had a positive experience. "
            "We appreciate you taking the time to let us know and hope to serve you again soon."
        )

    return (
        "Thank you for sharing your feedback. "
        f"We understand that the main point may involve: {issue_summary}. "
        "We appreciate the details and would be happy to look into this further if you share more information."
    )


# ------------------------------------------------------------
# 8. Detect bad generation
# ------------------------------------------------------------

def is_bad_generation(text):
    """
    Detect clearly unusable generations.

    This guard is intentionally conservative. We only fall back when the model
    output is clearly bad, because the goal is to use the generative model
    whenever the response is reasonably acceptable.
    """

    lowered = text.lower().strip()

    bad_phrases = [
        "i would not recommend",
        "main issues:",
        "predicted sentiment:",
        "customer message:",
        "write a helpful",
        "sentence 1:",
        "sentence 2:",
        "sentence 3:",
        "classifier",
        "faiss",
        "retrieval",
        "similar reviews",
        "[customer",
        "[your",
        "[company",
        "dear [",
    ]

    # Too short to be useful.
    if len(lowered.split()) < 10:
        return True

    # Prompt leakage or obvious review-style failure.
    if any(phrase in lowered for phrase in bad_phrases):
        return True

    return False


# ------------------------------------------------------------
# 9. Generate contextual response
# ------------------------------------------------------------

def generate_contextual_response(
    user_query,
    predicted_sentiment,
    retrieved_reviews,
    generator_tokenizer,
    generator_model,
    device,
    max_new_tokens=140,
):
    """
    Generate a customer-support-style response using a local chat model.

    Returns:
        A dictionary containing:
            - prompt
            - raw_generated_response
            - cleaned_generated_response
            - response
            - used_fallback
            - retrieved_context_for_display
    """

    messages = build_generation_messages(
        user_query=user_query,
        predicted_sentiment=predicted_sentiment,
        retrieved_reviews=retrieved_reviews,
    )

    prompt = generator_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = generator_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = generator_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.4,
            top_p=0.9,
            repetition_penalty=1.15,
            pad_token_id=generator_tokenizer.eos_token_id,
        )

    # For causal language models, the output includes the prompt plus the new text.
    # We slice off the prompt tokens and decode only the generated continuation.
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]

    raw_generated_text = generator_tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    cleaned_generated_text = clean_generated_response(raw_generated_text)

    used_fallback = False

    if is_bad_generation(cleaned_generated_text):
        final_response = build_safe_fallback_response(
            user_query=user_query,
            predicted_sentiment=predicted_sentiment,
            retrieved_reviews=retrieved_reviews,
        )
        used_fallback = True
    else:
        final_response = cleaned_generated_text

    return {
        "prompt": prompt,
        "raw_generated_response": raw_generated_text,
        "cleaned_generated_response": cleaned_generated_text,
        "response": final_response,
        "used_fallback": used_fallback,
        "retrieved_context_for_display": build_retrieved_context_for_display(retrieved_reviews),
    }


# ------------------------------------------------------------
# 10. Manual test
# ------------------------------------------------------------

if __name__ == "__main__":

    fake_retrieved_reviews = [
        {
            "score": 0.8123,
            "sentiment": "negative",
            "text": "The food arrived cold and the delivery was much later than expected. Nobody from the restaurant answered when I called.",
        },
        {
            "score": 0.7841,
            "sentiment": "negative",
            "text": "The service was disappointing and the order took forever. I would not recommend this place.",
        },
        {
            "score": 0.7022,
            "sentiment": "neutral",
            "text": "The food was okay, but the wait time was longer than expected.",
        },
    ]

    tokenizer, model, device = load_generator()

    result = generate_contextual_response(
        user_query="The food arrived cold and the delivery was very late.",
        predicted_sentiment="negative",
        retrieved_reviews=fake_retrieved_reviews,
        generator_tokenizer=tokenizer,
        generator_model=model,
        device=device,
    )

    print("=" * 80)
    print("PROMPT")
    print("=" * 80)
    print(result["prompt"])

    print()
    print("=" * 80)
    print("RAW GENERATED RESPONSE")
    print("=" * 80)
    print(result["raw_generated_response"])

    print()
    print("=" * 80)
    print("CLEANED GENERATED RESPONSE")
    print("=" * 80)
    print(result["cleaned_generated_response"])

    print()
    print("=" * 80)
    print("FINAL RESPONSE")
    print("=" * 80)
    print(result["response"])

    print()
    print("=" * 80)
    print("USED FALLBACK?")
    print("=" * 80)
    print(result["used_fallback"])

    print()
    print("=" * 80)
    print("RETRIEVED CONTEXT FOR DISPLAY")
    print("=" * 80)

    for item in result["retrieved_context_for_display"]:
        print()
        print(f"Rank: {item['rank']}")
        print(f"Score: {item['score']:.3f}")
        print(f"Sentiment: {item['sentiment']}")
        print(f"Text: {item['text']}")