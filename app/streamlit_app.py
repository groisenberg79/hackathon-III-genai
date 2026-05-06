import sys
from pathlib import Path

import streamlit as st


# ------------------------------------------------------------
# 1. Make src/ importable
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


from sentiment_classifier import load_lora_classifier, predict_sentiment, predict_sentiment_with_scores
from retriever_faiss import FaissReviewRetriever
from generate_response_remote import generate_contextual_response


# ------------------------------------------------------------
# 2. Streamlit page configuration
# ------------------------------------------------------------

st.set_page_config(
    page_title="Sentiment-Aware RAG Assistant",
    page_icon="💬",
    layout="wide",
)


# ------------------------------------------------------------
# 3. Cached loading
# ------------------------------------------------------------

@st.cache_resource
def load_classifier_cached():
    """
    Load the LoRA sentiment classifier once.

    Streamlit reruns this script whenever the user interacts with the app.
    Without caching, the classifier would reload on every button click.
    """

    return load_lora_classifier()


@st.cache_resource
def load_retriever_cached():
    """
    Load the FAISS retriever once.

    This loads:
        - MiniLM embedding model
        - FAISS index
        - review metadata
    """

    return FaissReviewRetriever()


# ------------------------------------------------------------
# 4. Helper functions
# ------------------------------------------------------------

def sentiment_badge(sentiment):
    """
    Display sentiment with appropriate Streamlit styling.
    """

    if sentiment == "negative":
        st.error("Negative")
    elif sentiment == "neutral":
        st.info("Neutral")
    elif sentiment == "positive":
        st.success("Positive")
    else:
        st.warning(sentiment)


def display_score_table(scores):
    """
    Display classifier probabilities in a compact table.
    """

    rows = [
        {
            "Sentiment": label,
            "Score": round(score, 4),
        }
        for label, score in scores.items()
    ]

    st.dataframe(
        rows,
        use_container_width=True,
        hide_index=True,
    )


def display_retrieved_reviews(retrieved_reviews):
    """
    Display retrieved reviews in expandable sections.
    """

    for i, review in enumerate(retrieved_reviews, start=1):
        title = (
            f"Review {i} | "
            f"Sentiment: {review['sentiment']} | "
            f"Similarity: {review['score']:.3f}"
        )

        with st.expander(title):
            st.write(review["text"])
            st.caption(f"Original Yelp label: {review['label']}")


# ------------------------------------------------------------
# 5. App header
# ------------------------------------------------------------

st.title("💬 Sentiment-Aware RAG Response Assistant")

st.write(
    """
This app analyzes customer feedback, retrieves semantically similar Yelp reviews,
and generates a context-aware customer support response.
"""
)

st.markdown(
    """
**Pipeline:** LoRA fine-tuned DistilBERT → FAISS retrieval with sentiment filtering → Llama response generation through OpenRouter.
"""
)


# ------------------------------------------------------------
# 6. Sidebar
# ------------------------------------------------------------

with st.sidebar:
    st.header("About this demo")

    st.markdown(
        """
This hackathon project combines:

1. **Sentiment classification**  
   A LoRA fine-tuned DistilBERT model predicts whether the customer message is negative, neutral, or positive.

2. **Semantic retrieval**  
   FAISS retrieves similar Yelp reviews from a local review index.

3. **Sentiment-aware filtering**  
   Retrieved reviews are filtered so that the context matches both the topic and sentiment of the user message.

4. **Generative response**  
   A Llama model accessed through OpenRouter writes a customer-support-style response.
"""
    )

    st.divider()

    st.markdown("**Generator model**")
    st.code("meta-llama/llama-3.1-8b-instruct")

    st.markdown("**Retriever**")
    st.code("sentence-transformers/all-MiniLM-L6-v2 + FAISS")

    st.markdown("**Classifier**")
    st.code("DistilBERT + LoRA adapter")


# ------------------------------------------------------------
# 7. Example inputs
# ------------------------------------------------------------

example_queries = {
    "Negative example": "The food arrived cold and the delivery took forever. Nobody answered when I called.",
    "Neutral example": "The meal was fine, nothing amazing, but the staff were polite.",
    "Positive example": "The food was delicious, the service was friendly, and everything arrived quickly.",
}

selected_example = st.selectbox(
    "Choose an example or write your own:",
    ["Custom input"] + list(example_queries.keys()),
)

if selected_example == "Custom input":
    default_text = ""
else:
    default_text = example_queries[selected_example]


user_query = st.text_area(
    "Customer message",
    value=default_text,
    height=150,
    placeholder="Type a customer review, complaint, or feedback message here...",
)


col_settings_1, col_settings_2 = st.columns([1, 1])

with col_settings_1:
    top_k = st.slider(
        "Number of similar reviews to show",
        min_value=1,
        max_value=5,
        value=3,
    )

with col_settings_2:
    candidate_k = st.slider(
        "Candidate reviews searched before sentiment filtering",
        min_value=5,
        max_value=50,
        value=20,
        step=5,
    )


run_button = st.button(
    "Analyze and generate response",
    type="primary",
)


# ------------------------------------------------------------
# 8. Main app logic
# ------------------------------------------------------------

if run_button:
    if not user_query.strip():
        st.warning("Please enter a customer message first.")

    else:
        with st.spinner("Loading classifier and retriever..."):
            classifier_tokenizer, classifier_model, classifier_device = load_classifier_cached()
            retriever = load_retriever_cached()

        with st.spinner("Predicting sentiment..."):
            prediction_result = predict_sentiment_with_scores(
                text=user_query,
                tokenizer=classifier_tokenizer,
                model=classifier_model,
                device=classifier_device,
            )

            predicted_sentiment = prediction_result["label"]
            sentiment_scores = prediction_result["scores"]

        with st.spinner("Retrieving similar reviews..."):
            retrieved_reviews = retriever.retrieve_with_sentiment_filter(
                query=user_query,
                sentiment=predicted_sentiment,
                top_k=top_k,
                candidate_k=candidate_k,
            )

        with st.spinner("Generating response with OpenRouter..."):
            generation_result = generate_contextual_response(
                user_query=user_query,
                predicted_sentiment=predicted_sentiment,
                retrieved_reviews=retrieved_reviews,
            )

        st.divider()

        # --------------------------------------------------------
        # Results summary
        # --------------------------------------------------------

        left_col, right_col = st.columns([1, 2])

        with left_col:
            st.subheader("Predicted Sentiment")
            sentiment_badge(predicted_sentiment)

            st.caption(f"Classifier device: {classifier_device}")

            st.subheader("Classifier Scores")
            display_score_table(sentiment_scores)

        with right_col:
            st.subheader("Generated Customer Support Response")
            st.write(generation_result["response"])

            st.caption(f"Generated with: {generation_result['model']}")

        st.divider()

        # --------------------------------------------------------
        # Retrieved reviews
        # --------------------------------------------------------

        st.subheader("Retrieved Similar Reviews")

        st.write(
            """
These are reviews retrieved by FAISS and filtered to match the predicted sentiment.
They provide contextual examples for the app's retrieval step.
"""
        )

        display_retrieved_reviews(retrieved_reviews)

        st.divider()

        # --------------------------------------------------------
        # Debug information
        # --------------------------------------------------------

        with st.expander("Debug: raw model response"):
            st.write(generation_result["raw_generated_response"])

        with st.expander("Debug: messages sent to OpenRouter"):
            st.json(generation_result["prompt"])