# ------------------------------------------------------------
# Sentiment calibration
# ------------------------------------------------------------

def calibrate_sentiment(user_query, predicted_sentiment):
    """
    Light correction layer for obvious sentiment cases.

    The LoRA classifier is useful, but not perfect. This function catches
    simple, obvious cases where the text itself strongly suggests positive,
    neutral, or negative sentiment.

    Priority:
        1. Strong negative evidence -> negative
        2. Neutral/mixed evidence -> neutral
        3. Clear positive evidence -> positive
        4. Otherwise trust the model
    """

    text = user_query.lower()

    positive_words = [
        "delicious",
        "friendly",
        "quickly",
        "great",
        "excellent",
        "amazing",
        "loved",
        "perfect",
        "fresh",
        "tasty",
        "wonderful",
        "fast",
    ]

    neutral_phrases = [
        "fine",
        "nothing amazing",
        "nothing special",
        "okay",
        "average",
        "polite",
        "decent",
        "acceptable",
        "not bad",
        "not great",
        "mixed",
    ]

    negative_phrases = [
        "nobody answered",
        "no one answered",
        "wrong order",
        "missing item",
        "incorrect order",
    ]

    negative_words = [
        "cold",
        "late",
        "forever",
        "ignored",
        "rude",
        "terrible",
        "awful",
        "horrible",
        "missing",
        "disappointed",
        "bad",
    ]

    has_positive = any(word in text for word in positive_words)
    has_neutral = any(phrase in text for phrase in neutral_phrases)

    has_negative_phrase = any(phrase in text for phrase in negative_phrases)

    # Avoid treating "not bad" as negative just because it contains "bad".
    text_for_negative_words = text.replace("not bad", "")

    has_negative_word = any(word in text_for_negative_words for word in negative_words)

    has_negative = has_negative_phrase or has_negative_word

    # Strong negative evidence wins.
    if has_negative:
        return "negative"

    # Neutral/mixed language should override mild positive words like "polite".
    # Example: "fine, nothing amazing, but staff were polite" should be neutral.
    if has_neutral:
        return "neutral"

    # Clear positive evidence with no negative or neutral evidence.
    if has_positive:
        return "positive"

    # Otherwise trust the model.
    return predicted_sentiment


# ------------------------------------------------------------
# Manual test
# ------------------------------------------------------------

if __name__ == "__main__":

    test_cases = [
        {
            "text": "The food arrived cold and the delivery took forever.",
            "raw_prediction": "negative",
        },
        {
            "text": "The meal was fine, nothing amazing, but the staff were polite.",
            "raw_prediction": "negative",
        },
        {
            "text": "The food was delicious, the service was friendly, and everything arrived quickly.",
            "raw_prediction": "positive",
        },
        {
            "text": "The food was not bad, but nothing special.",
            "raw_prediction": "negative",
        },
    ]

    for case in test_cases:
        calibrated = calibrate_sentiment(
            user_query=case["text"],
            predicted_sentiment=case["raw_prediction"],
        )

        print("=" * 80)
        print(case["text"])
        print(f"Raw prediction: {case['raw_prediction']}")
        print(f"Calibrated sentiment: {calibrated}")