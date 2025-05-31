import streamlit as st
import joblib
import re
import torch
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

from transformers import PegasusTokenizer, PegasusForConditionalGeneration, pipeline

# ── Load Models and Tools ─────────────────
nb_model = joblib.load("naive_bayes_model.pkl")
lr_model = joblib.load("logistic_regression_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
lstm_model = load_model("lstm_model.keras")

with open("lstm_tokenizer.json") as f:
    keras_tokenizer = tokenizer_from_json(f.read())

hf_tokenizer = DistilBertTokenizerFast.from_pretrained("bert_model/")
hf_model = DistilBertForSequenceClassification.from_pretrained("bert_model/")
hf_model.eval()

le = joblib.load("label_encoder.pkl")

# ── Pegasus Summarizer ────────────────────
model_name = "google/pegasus-cnn_dailymail"
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name)
# Choose device (0 = GPU, -1 = CPU)
import torch
device = 0 if torch.cuda.is_available() else -1

# Initialize pipeline
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer, device=device)

# Movie Description Cleaning
import re, string, nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm.auto import tqdm
import contractions

nltk.download('stopwords')

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
punct_tbl  = str.maketrans("", "", string.punctuation)


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = contractions.fix(text) # "doesn't" → "does not"
    text = text.strip().lower() # convert to lower case letters
    text = re.sub(r"[^a-z\s]", "", text) # keep only English letters
    text = re.sub(r"(https?://\S+|www\.\S+|\S+@\S+)", " ", text)  # drop URLs & e-mails
    text = re.sub(r"\b\d+\b", " ", text)                          # replace standalone numbers with space
    text = text.translate(punct_tbl)                              # punctuation

    # drop stop words
    words = text.split()
    words = [w for w in words if w not in stop_words]
    cleaned = " ".join(words)

    return re.sub(r"\s{2,}", " ", cleaned).strip()


def chunk_text(text, chunk_size=500):
    """
    Split long text into smaller chunks of ~chunk_size words.
    """
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def summarize_long_text(text, summarizer, max_length=200, min_length=70):
    """
    Break long text into chunks, summarize each chunk, and return concatenated summaries.
    """
    chunks = chunk_text(text, chunk_size=500)
    summaries = []

    for chunk in chunks:
        try:
            summary = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
            summaries.append(summary)
        except Exception as e:
            print(f"[Error] Skipping chunk due to: {e}")
            continue

    return " ".join(summaries)


# ── Prediction Function ───────────────────
def predict_genre(summary):
    cleaned = re.sub(r"[^\w\s]", "", summary.lower())

    vec = vectorizer.transform([cleaned])
    nb_result = le.inverse_transform(nb_model.predict(vec))[0]
    lr_result = le.inverse_transform(lr_model.predict(vec))[0]

    seq = keras_tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=300)
    lstm_pred = tf.argmax(lstm_model.predict(padded, verbose=0), axis=1).numpy()[0]
    lstm_result = le.inverse_transform([lstm_pred])[0]

    bert_inputs = hf_tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        bert_outputs = hf_model(**bert_inputs)
    bert_pred = torch.argmax(bert_outputs.logits, dim=1).item()
    bert_result = le.inverse_transform([bert_pred])[0]

    return {
        "Naive Bayes": nb_result,
        "Logistic Regression": lr_result,
        "LSTM": lstm_result,
        "Transformer (BERT)": bert_result
    }

# ── Streamlit UI ──────────────────────────
st.set_page_config(page_title="Movie Analysis Tool", layout="centered")
st.title("Movie Analysis Tool: Plot Summary and Genre Predictor")

# Step 1: Text input
text_input = st.text_area("Enter a movie description or plot:")

# Step 2: Single action button
if st.button("Analyze"):
    if text_input.strip() == "":
        st.warning("Please enter a description before analyzing.")
    else:
        with st.spinner("Summarizing..."):
            final_input = summarize_long_text(text_input, summarizer)
        st.success("Summarization Complete!")

        st.subheader("Summarized Movie Description")
        st.write(final_input)

        # Clean the summarized description
        cleaned_description = clean_text(final_input)
        st.subheader("Cleaned Movie Description")
        st.write(cleaned_description)

        with st.spinner("Predicting genre..."):
            results = predict_genre(final_input)
        st.success("Prediction Complete!")

        st.subheader("Genre Predictions")
        for model, genre in results.items():
            st.write(f"**{model}**: {genre}")