# Sentiment-Aware RAG Response Assistant

A GenAI hackathon project that combines sentiment classification, semantic retrieval, and LLM-based response generation to produce context-aware customer support replies.

The app takes a customer message, predicts whether the sentiment is negative, neutral, or positive, retrieves semantically similar Yelp reviews, filters them by sentiment, and then uses a Llama model through OpenRouter to generate a professional customer-support-style response.

---

## Project Overview

This project was built for a GenAI hackathon with the objective of creating a parameter-efficient fine-tuned model for sentiment analysis and combining it with retrieval-based context and generative responses.

The final system uses:

- **LoRA fine-tuned DistilBERT** for sentiment classification
- **FAISS** for local semantic search over Yelp reviews
- **SentenceTransformer embeddings** for review retrieval
- **Sentiment-aware retrieval filtering** to keep retrieved examples aligned with the predicted sentiment
- **OpenRouter + Llama 3.1 8B Instruct** for response generation
- **Streamlit** for the user interface

---

## Demo Workflow

The app supports three types of customer feedback.

### Negative feedback

Input:

```text
The food arrived cold and the delivery took forever. Nobody answered when I called.
```

Expected behavior:

- Predicts negative sentiment
- Retrieves similar negative Yelp reviews
- Generates an apology and asks for order details so the team can follow up

### Neutral feedback

Input:

```text
The meal was fine, nothing amazing, but the staff were polite.
```

Expected behavior:

- Predicts neutral sentiment
- Retrieves similar neutral Yelp reviews
- Generates a balanced response asking what could be improved

### Positive feedback

Input:

```text
The food was delicious, the service was friendly, and everything arrived quickly.
```

Expected behavior:

- Predicts positive sentiment
- Retrieves similar positive Yelp reviews
- Generates an appreciative response thanking the customer

---

## Architecture

```text
Customer message
      ↓
LoRA fine-tuned DistilBERT classifier
      ↓
Predicted sentiment
      ↓
FAISS semantic retrieval
      ↓
Sentiment-aware filtering
      ↓
Similar Yelp reviews
      ↓
Prompt construction
      ↓
OpenRouter / Llama 3.1 8B Instruct
      ↓
Generated customer support response
```

---

## Main Components

### 1. Sentiment Classifier

File:

```text
src/sentiment_classifier.py
```

The classifier uses `distilbert-base-uncased` with a LoRA adapter trained on a subset of the Yelp Review Full dataset.

The original Yelp labels are mapped as follows:

```text
1–2 stars → negative
3 stars   → neutral
4–5 stars → positive
```

The final demo model was trained on a random sample of 10,000 Yelp reviews and saved as a LoRA adapter.

---

### 2. FAISS Retriever

File:

```text
src/retriever_faiss.py
```

The retriever uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

to create embeddings for Yelp reviews and stores them in a local FAISS index.

Given a new customer message, the retriever finds semantically similar reviews.

The retrieval step also includes sentiment-aware filtering:

```text
1. Retrieve candidate reviews with FAISS
2. Keep reviews that match the predicted sentiment
3. Return the top matching examples
```

This avoids cases where a topically similar but emotionally opposite review is retrieved.

For example, a complaint about cold food and late delivery should not retrieve a review praising fast delivery and warm food.

---

### 3. Remote LLM Response Generation

File:

```text
src/generate_response_remote.py
```

The response generator uses OpenRouter to call:

```text
meta-llama/llama-3.1-8b-instruct
```

The model receives:

- the original customer message
- the predicted sentiment
- a compact summary of the user’s message
- retrieved review context for display/debugging

The prompt is branched by sentiment:

- negative messages receive apology/follow-up style responses
- neutral messages receive balanced acknowledgement responses
- positive messages receive appreciation responses

---

### 4. Streamlit App

File:

```text
app/streamlit_app.py
```

The Streamlit app provides a simple interface where users can:

- choose a sample customer message or write their own
- run sentiment analysis
- view classifier probabilities
- view retrieved similar Yelp reviews
- generate a customer support response

---

## Project Structure

```text
hackathon-III-genai/
├── app/
│   └── streamlit_app.py
│
├── src/
│   ├── build_faiss_index.py
│   ├── generate_response_remote.py
│   ├── retriever_faiss.py
│   ├── sentiment_calibration.py
│   ├── sentiment_classifier.py
│   ├── test_pipeline.py
│   └── train_classifier.py
│
├── models/
│   ├── sentiment_classifier_lora_random_10k/
│   └── faiss_index/
│
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

Note: the `models/` directory is not committed to GitHub because it contains generated model and index files.

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/groisenberg79/hackathon-III-genai.git
cd hackathon-III-genai
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Create a `.env` file

Create a file named `.env` in the project root:

```bash
touch .env
```

Add your OpenRouter credentials:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=meta-llama/llama-3.1-8b-instruct
```

Do not commit `.env` to GitHub.

A safe example file is provided:

```text
.env.example
```

---

## Running the Project

### 1. Train the LoRA sentiment classifier

```bash
python src/train_classifier.py
```

This trains the LoRA classifier and saves it under:

```text
models/sentiment_classifier_lora_random_10k/
```

### 2. Build the FAISS index

```bash
python src/build_faiss_index.py
```

This creates:

```text
models/faiss_index/reviews.index
models/faiss_index/review_metadata.pkl
```

### 3. Test individual components

Test the classifier:

```bash
python src/sentiment_classifier.py
```

Test the retriever:

```bash
python src/retriever_faiss.py
```

Test the remote generator:

```bash
python src/generate_response_remote.py
```

Test the full pipeline:

```bash
python src/test_pipeline.py
```

### 4. Run the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

Then open the local Streamlit URL in your browser.

Usually:

```text
http://localhost:8501
```

---

## Environment Variables

The app requires the following environment variables:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=meta-llama/llama-3.1-8b-instruct
```

The OpenRouter API is only called when generating a response. Local steps such as loading FAISS, predicting sentiment, and selecting examples do not consume OpenRouter credits.

---

## Requirements

Main libraries used:

```text
torch
transformers
peft
datasets
scikit-learn
sentence-transformers
faiss-cpu
streamlit
requests
python-dotenv
numpy
```

---

## Model Training Notes

Several training strategies were tested.

### Balanced 10k LoRA training

A balanced dataset was created with equal numbers of negative, neutral, and positive examples.

This improved class balance conceptually, but validation performance decreased.

### Random 10k LoRA training

A random sample of 10,000 Yelp reviews was used for the final demo model.

Although its validation metrics were not perfect, it performed well on the target demo cases:

```text
negative complaint → negative
mixed/moderate feedback → neutral
positive review → positive
```

This version is used in the final app.

---

## Retrieval Design Notes

The first retrieval version used plain FAISS similarity search.

However, this sometimes retrieved reviews that were topically similar but had the opposite emotional tone.

For example:

```text
Query: cold food and late delivery
Bad retrieval: fast delivery and warm food
```

To improve this, the final retriever uses sentiment-aware filtering:

```text
1. Retrieve more candidates than needed
2. Filter candidates by predicted sentiment
3. Return the top matching reviews
```

This produces more coherent context for the final response generation step.

---

## Known Limitations

This is a hackathon prototype, not a production system.

Current limitations:

- The sentiment classifier is trained on a limited sample of Yelp reviews.
- The FAISS index is built from a sampled subset of the Yelp dataset, not the full dataset.
- The response generator depends on OpenRouter availability.
- The app currently runs locally and is not deployed to a public server.
- The generated responses should be reviewed before being used in real customer support workflows.

---

## Possible Future Improvements

Potential next steps:

- Deploy the app publicly with Streamlit Community Cloud, Render, or a VPS
- Move vector search from FAISS to Pinecone for hosted retrieval
- Add a FastAPI backend
- Improve LoRA training with better validation and hyperparameter tuning
- Add user feedback buttons to collect ratings on generated responses
- Store feedback for future fine-tuning
- Add latency and cost tracking for generated responses
- Compare multiple generator models through OpenRouter

---

## Hackathon Summary

This project demonstrates a practical GenAI pipeline:

```text
fine-tuned classifier
+ retrieval
+ LLM response generation
+ simple user interface
```

The goal was not only to classify sentiment, but to use sentiment and retrieved context to produce useful, tone-appropriate customer support replies.

The final app shows how a lightweight fine-tuned model and a retrieval component can work together with a stronger hosted LLM to create an end-to-end sentiment-aware response assistant.