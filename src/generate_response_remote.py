import os
import re
import time

import requests
from dotenv import load_dotenv


# ------------------------------------------------------------
# 1. Environment setup
# ------------------------------------------------------------

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

OPENROUTER_MODEL = os.getenv(
    "OPENROUTER_MODEL",
    "meta-llama/llama-3.1-8b-instruct",
)


# ------------------------------------------------------------
# 2. Infer summary from the user's own message
# ------------------------------------------------------------

def infer_user_summary(user_query, predicted_sentiment):
    """
    Build a compact summary from the user's own message.

    Important:
    We intentionally prioritize the user query over retrieved reviews.

    Retrieved reviews are useful for showing similar cases, but they can also
    introduce problems the current user did not actually mention. For example,
    a retrieved review might mention a long wait even when the user's message
    was positive. So this function only looks at the user's actual message.
    """

    text = user_query.lower()

    negative_issues = []
    positive_points = []
    neutral_points = []

    # Negative issue detection
    if any(word in text for word in ["cold", "temperature", "lukewarm"]):
        negative_issues.append("food arrived cold")

    if any(word in text for word in ["late", "delay", "delayed", "forever", "slow", "waiting"]):
        negative_issues.append("delivery or service was delayed")

    if any(
        phrase in text
        for phrase in [
            "nobody answered",
            "no one answered",
            "did not answer",
            "ignored",
            "could not reach",
        ]
    ):
        negative_issues.append("customer could not reach support")

    if any(word in text for word in ["rude", "disrespectful", "unfriendly"]):
        negative_issues.append("staff interaction was disappointing")

    if any(
        phrase in text
        for phrase in [
            "wrong item",
            "missing item",
            "incorrect order",
            "wrong order",
        ]
    ):
        negative_issues.append("order accuracy problem")

    # Positive point detection
    if any(word in text for word in ["delicious", "tasty", "great", "excellent", "amazing", "good"]):
        positive_points.append("the food was enjoyable")

    if any(word in text for word in ["friendly", "polite", "kind", "helpful"]):
        positive_points.append("the staff or service was positive")

    if any(word in text for word in ["quickly", "fast", "promptly", "on time"]):
        positive_points.append("the order arrived quickly")

    # Neutral / mixed point detection
    if any(
        phrase in text
        for phrase in [
            "fine",
            "okay",
            "average",
            "nothing amazing",
            "nothing special",
            "not bad",
            "not great",
        ]
    ):
        neutral_points.append("the experience was acceptable but not especially memorable")

    if predicted_sentiment == "positive":
        if positive_points:
            return ", ".join(positive_points)
        return "customer had a positive experience"

    if predicted_sentiment == "neutral":
        if neutral_points:
            return ", ".join(neutral_points)
        if positive_points:
            return ", ".join(positive_points)
        return "customer shared moderate or mixed feedback"

    # Negative case
    if negative_issues:
        return ", ".join(negative_issues)

    return "customer had a disappointing experience"


# ------------------------------------------------------------
# 3. Retrieved context for display
# ------------------------------------------------------------

def build_retrieved_context_for_display(retrieved_reviews, max_reviews=3, max_chars=220):
    """
    Prepare retrieved reviews for display in the app.

    These snippets demonstrate the retrieval step, but they are not used to
    invent issues in the generated response.
    """

    snippets = []

    for i, review in enumerate(retrieved_reviews[:max_reviews], start=1):
        text = review["text"].replace("\n", " ").strip()

        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "..."

        snippets.append(
            {
                "rank": review.get("rank", i),
                "score": review["score"],
                "sentiment": review["sentiment"],
                "text": text,
            }
        )

    return snippets


# ------------------------------------------------------------
# 4. Build messages for remote LLM
# ------------------------------------------------------------

def build_generation_messages(user_query, predicted_sentiment):
    """
    Build chat messages for the OpenRouter model.

    We branch by sentiment because a positive message should not receive an
    apology or escalation request, while a negative message should.
    """

    user_summary = infer_user_summary(
        user_query=user_query,
        predicted_sentiment=predicted_sentiment,
    )

    system_message = """
You are a professional customer support assistant for a restaurant or delivery business.
Write replies from the business to the customer.
Do not write a customer review.
Do not mention AI, classifiers, embeddings, FAISS, retrieval, or similar reviews.
Do not invent a refund policy.
Do not add specific details that the customer did not provide.
Do not use placeholders.
Do not include greetings, signatures, or subject lines.
""".strip()

    if predicted_sentiment == "negative":
        user_message = f"""
Customer message:
{user_query}

Predicted sentiment:
negative

Main issues mentioned by the customer:
{user_summary}

Write only the final customer support reply.
Write exactly 3 concise sentences.
Sentence 1: apologize or acknowledge the negative experience.
Sentence 2: briefly mention the main issue.
Sentence 3: ask for order/visit details or offer to have the team follow up.
""".strip()

    elif predicted_sentiment == "positive":
        user_message = f"""
Customer message:
{user_query}

Predicted sentiment:
positive

Positive points mentioned by the customer:
{user_summary}

Write only the final customer support reply.
Write exactly 3 concise sentences.
Sentence 1: thank the customer for the positive feedback.
Sentence 2: briefly mention what they enjoyed.
Sentence 3: say that the team appreciates the feedback and looks forward to serving them again.
Do not apologize.
Do not ask for order details.
Do not imply that something went wrong.
""".strip()

    else:
        user_message = f"""
Customer message:
{user_query}

Predicted sentiment:
neutral

Main points mentioned by the customer:
{user_summary}

Write only the final customer support reply.
Write exactly 3 concise sentences.
Sentence 1: thank the customer for sharing their feedback.
Sentence 2: acknowledge that the experience sounded moderate or mixed.
Sentence 3: invite them to share more details about what could be improved.
Do not apologize strongly.
Do not invent a problem that was not mentioned.
""".strip()

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]


# ------------------------------------------------------------
# 5. Clean response
# ------------------------------------------------------------

def clean_generated_response(text):
    """
    Clean small formatting artifacts from model output.
    """

    text = text.strip()

    text = re.sub(
        r"^(assistant|customer support response|response)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(r"\s+", " ", text).strip()

    return text


# ------------------------------------------------------------
# 6. OpenRouter API call
# ------------------------------------------------------------

def call_openrouter(messages, temperature=0.25, max_tokens=180, max_retries=3):
    """
    Send messages to OpenRouter and return the generated response text.

    Retries are included because OpenRouter may route to providers that
    occasionally return temporary upstream errors.
    """

    if not OPENROUTER_API_KEY:
        raise ValueError(
            "OPENROUTER_API_KEY is missing. Add it to your .env file."
        )

    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "Hackathon Sentiment RAG App",
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=60,
            )

            try:
                data = response.json()
            except ValueError:
                raise RuntimeError(
                    f"OpenRouter returned a non-JSON response.\n"
                    f"Status code: {response.status_code}\n"
                    f"Response text: {response.text}"
                )

            if response.status_code != 200:
                raise RuntimeError(
                    f"OpenRouter API error.\n"
                    f"Status code: {response.status_code}\n"
                    f"Response JSON: {data}"
                )

            if "choices" not in data:
                raise RuntimeError(
                    f"OpenRouter response did not contain 'choices'.\n"
                    f"Response JSON: {data}"
                )

            return data["choices"][0]["message"]["content"]

        except Exception as error:
            last_error = error

            print(f"OpenRouter attempt {attempt}/{max_retries} failed:")
            print(error)

            if attempt < max_retries:
                time.sleep(2)

    raise RuntimeError(
        f"OpenRouter failed after {max_retries} attempts.\n"
        f"Last error: {last_error}"
    )


# ------------------------------------------------------------
# 7. Main generation function
# ------------------------------------------------------------

def generate_contextual_response(
    user_query,
    predicted_sentiment,
    retrieved_reviews,
):
    """
    Generate a context-aware customer support response using OpenRouter.
    """

    messages = build_generation_messages(
        user_query=user_query,
        predicted_sentiment=predicted_sentiment,
    )

    raw_response = call_openrouter(messages)

    cleaned_response = clean_generated_response(raw_response)

    return {
        "prompt": messages,
        "raw_generated_response": raw_response,
        "cleaned_generated_response": cleaned_response,
        "response": cleaned_response,
        "used_fallback": False,
        "model": OPENROUTER_MODEL,
        "retrieved_context_for_display": build_retrieved_context_for_display(
            retrieved_reviews
        ),
    }


# ------------------------------------------------------------
# 8. Manual test
# ------------------------------------------------------------

if __name__ == "__main__":

    fake_retrieved_reviews = [
        {
            "rank": 1,
            "score": 0.8123,
            "sentiment": "negative",
            "text": "The food arrived cold and the delivery was much later than expected. Nobody from the restaurant answered when I called.",
        },
        {
            "rank": 2,
            "score": 0.7841,
            "sentiment": "negative",
            "text": "The service was disappointing and the order took forever. I would not recommend this place.",
        },
    ]

    test_cases = [
        {
            "user_query": "The food arrived cold and the delivery was very late.",
            "predicted_sentiment": "negative",
        },
        {
            "user_query": "The meal was fine, nothing amazing, but the staff were polite.",
            "predicted_sentiment": "neutral",
        },
        {
            "user_query": "The food was delicious, the service was friendly, and everything arrived quickly.",
            "predicted_sentiment": "positive",
        },
    ]

    for test_case in test_cases:
        result = generate_contextual_response(
            user_query=test_case["user_query"],
            predicted_sentiment=test_case["predicted_sentiment"],
            retrieved_reviews=fake_retrieved_reviews,
        )

        print("=" * 80)
        print("USER QUERY")
        print("=" * 80)
        print(test_case["user_query"])

        print()
        print("=" * 80)
        print("PREDICTED SENTIMENT")
        print("=" * 80)
        print(test_case["predicted_sentiment"])

        print()
        print("=" * 80)
        print("MODEL")
        print("=" * 80)
        print(result["model"])

        print()
        print("=" * 80)
        print("FINAL RESPONSE")
        print("=" * 80)
        print(result["response"])

        print()
        print("#" * 80)
        print()