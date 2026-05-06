from collections import Counter


# ------------------------------------------------------------
# 1. Helper function: summarize retrieved sentiments
# ------------------------------------------------------------

def summarize_retrieved_sentiments(retrieved_reviews):
    """
    Count the sentiment labels among the retrieved reviews.

    Example:
        retrieved sentiments = ["negative", "negative", "neutral"]

    Output:
        "Most similar reviews were mostly negative."
    """

    if not retrieved_reviews:
        return "No similar reviews were retrieved."

    sentiments = [review["sentiment"] for review in retrieved_reviews]

    sentiment_counts = Counter(sentiments)

    most_common_sentiment, count = sentiment_counts.most_common(1)[0]

    total = len(retrieved_reviews)

    return (
        f"Among the {total} most similar retrieved reviews, "
        f"the most common sentiment was '{most_common_sentiment}' "
        f"({count}/{total} reviews)."
    )


# ------------------------------------------------------------
# 2. Helper function: extract short context snippets
# ------------------------------------------------------------

def build_context_snippets(retrieved_reviews, max_snippets=3, max_chars=220):
    """
    Create short readable snippets from retrieved reviews.

    These snippets are used as context for the generated response.
    """

    snippets = []

    for i, review in enumerate(retrieved_reviews[:max_snippets], start=1):
        text = review["text"].replace("\n", " ").strip()

        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "..."

        snippet = (
            f"{i}. Sentiment: {review['sentiment']} | "
            f"Similarity score: {review['score']:.3f} | "
            f"Review: {text}"
        )

        snippets.append(snippet)

    return "\n".join(snippets)


# ------------------------------------------------------------
# 3. Main response-generation function
# ------------------------------------------------------------

def generate_contextual_response(user_query, predicted_sentiment, retrieved_reviews):
    """
    Generate a context-aware response using:
        - the user's original query
        - the model's predicted sentiment
        - similar reviews retrieved with FAISS

    This version uses a deterministic template-based approach.
    """

    retrieved_summary = summarize_retrieved_sentiments(retrieved_reviews)
    context_snippets = build_context_snippets(retrieved_reviews)

    if predicted_sentiment == "negative":
        response = f"""
I'm sorry to hear about this experience. The message was classified as **negative**, which suggests the user is frustrated or dissatisfied.

{retrieved_summary}

Based on the retrieved examples, a good response should:
- acknowledge the problem directly,
- avoid sounding defensive,
- apologize for the negative experience,
- offer a concrete next step, such as follow-up, refund, replacement, or escalation.

Suggested response:

"Thank you for sharing this. I'm sorry the experience did not meet expectations. Based on what you described, the main issue seems to involve a disappointing service or product experience. We would like to look into this and help make it right. Please share any relevant order or visit details so the support team can follow up with an appropriate solution."
""".strip()

    elif predicted_sentiment == "neutral":
        response = f"""
Thanks for sharing this feedback. The message was classified as **neutral**, which suggests the user may be describing a mixed or average experience rather than a strongly positive or negative one.

{retrieved_summary}

Based on the retrieved examples, a good response should:
- acknowledge the feedback,
- ask for more detail if needed,
- avoid over-apologizing,
- invite the user to explain what could be improved.

Suggested response:

"Thank you for your feedback. It sounds like your experience was mixed or did not strongly stand out in either direction. We appreciate the details you shared and would be happy to learn more about what worked well and what could be improved. Your comments help us understand how to provide a better experience in the future."
""".strip()

    elif predicted_sentiment == "positive":
        response = f"""
Thank you for sharing this positive feedback. The message was classified as **positive**, which suggests the user had a satisfying or enjoyable experience.

{retrieved_summary}

Based on the retrieved examples, a good response should:
- thank the user warmly,
- reinforce the positive aspects,
- sound appreciative but not exaggerated,
- invite the user to return or continue engaging.

Suggested response:

"Thank you for the kind feedback. We're glad to hear that you had a positive experience. Comments like this are encouraging for the team, and we appreciate you taking the time to share them. We hope to have the opportunity to serve you again soon."
""".strip()

    else:
        response = f"""
The predicted sentiment was **{predicted_sentiment}**, which is not one of the expected labels: negative, neutral, or positive.

{retrieved_summary}

Suggested response:

"Thank you for sharing your feedback. We appreciate the details and will use them to better understand the experience and identify any useful next steps."
""".strip()

    return {
        "predicted_sentiment": predicted_sentiment,
        "retrieved_summary": retrieved_summary,
        "context_snippets": context_snippets,
        "response": response,
    }


# ------------------------------------------------------------
# 4. Manual test
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

    result = generate_contextual_response(
        user_query="The food arrived cold and the delivery was very late.",
        predicted_sentiment="negative",
        retrieved_reviews=fake_retrieved_reviews,
    )

    print("=" * 80)
    print("CONTEXT SNIPPETS")
    print("=" * 80)
    print(result["context_snippets"])

    print()
    print("=" * 80)
    print("GENERATED RESPONSE")
    print("=" * 80)
    print(result["response"])