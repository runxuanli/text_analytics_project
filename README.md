**Northwestern University — Text Analytics Final Project**  
**Spring 2025**

## 📌 Overview

This project applies multimodal machine learning techniques to classify movie genres using both textual and visual data from the [IMDB Multimodal Genre Dataset](https://www.kaggle.com/datasets/zulkarnainsaurav/imdb-multimodal-vision-and-nlp-genre-classification). Our models analyze movie poster images and plot summaries to predict genres: **Action**, **Comedy**, **Horror**, and **Romance**.

We implemented a full pipeline covering:
- Data cleaning and preprocessing
- Text summarization
- Classical and deep learning models
- Model evaluation
- Word importance visualization
- Interactive genre prediction tool

## Project Structure

text_analytics_project/
│
├── genre_app.py # Streamlit app for interactive predictions
├── text_project.ipynb # Main Jupyter notebook with code and analysis
├── label_encoder.pkl # Saved label encoder
├── tfidf_vectorizer.pkl # TF-IDF vectorizer
├── naive_bayes_model.pkl # Naive Bayes model
├── logistic_regression_model.pkl # Logistic Regression model
├── lstm_model.keras # Trained LSTM model
├── lstm_tokenizer.json # Tokenizer used for LSTM
├── lstm_model_architecture.png # LSTM model architecture plot
├── Top 20 Words/ # CSVs and charts of top words per genre
├── wordclouds/ # Word cloud visualizations
├── Project Discussion.html # HTML export of team discussions
└── README.md # This file


## Tasks Completed

1. **Data Preprocessing**
   - Cleaned and tokenized plot summaries
   - Removed noise (punctuation, stopwords, etc.)

2. **Summarization**
   - Built a chunking + summarizer pipeline using Hugging Face Transformers

3. **Modeling**
   - Naive Bayes (baseline)
   - Logistic Regression
   - LSTM (Keras)
   - BERT (DistilBERT from Hugging Face) *(optional/advanced)*

4. **Evaluation**
   - Accuracy by genre and overall
   - Plots for LSTM model accuracy vs. epochs

5. **Explainability**
   - Extracted top N words per genre
   - Created word clouds per model/genre

6. **Error Analysis**
   - Highlighted cases where models disagreed
   - Interpreted model decisions

7. **Interactive Tool**
   - Built with Streamlit to:
     - Input summaries
     - Clean text
     - Predict genre using all models

8. **Bonus (Visual Classification)**
   - Paired posters with genres
   - Optional CNN image classification (in progress)
  
## launch the interactive tool

streamlit run genre_app.py


