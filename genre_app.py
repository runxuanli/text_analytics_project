import streamlit as st
import joblib
import re
import torch
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

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
st.set_page_config(page_title="Genre Predictor", layout="centered")
st.title("🎬 Movie Genre Predictor")

summary = st.text_area("Enter a movie description below:")

if st.button("Predict"):
    if summary.strip() == "":
        st.warning("Please enter a summary.")
    else:
        with st.spinner("Predicting..."):
            results = predict_genre(summary)
        st.success("Prediction Complete!")

        for model, genre in results.items():
            st.write(f"**{model}**: {genre}")