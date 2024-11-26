import streamlit as st
import pandas as pd
import joblib
import os
import sys


# الحصول على المسار الكامل لمجلد src
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))

# إضافة المسار إلى sys.path
if src_path not in sys.path:
    sys.path.append(src_path)

from train_model import preprocess_text


def load_model():
  try:
      model = joblib.load('model/spam_model.joblib')
      vectorizer = joblib.load('model/vectorizer.joblib')
      return model, vectorizer
  except FileNotFoundError:
      st.error("Model files not found. Please train the model first.")
      return None, None

def predict_spam(text, model, vectorizer):
  processed_text = preprocess_text(text)
  text_vector = vectorizer.transform([processed_text])
  prediction = model.predict(text_vector)[0]
  probability = model.predict_proba(text_vector)[0]

  return {
      'is_spam': bool(prediction),
      'confidence': float(probability[1]) if prediction else float(probability[0])
  }

def main():
  st.set_page_config(
      page_title="Email Spam Detector",
      page_icon="📧",
      layout="wide"
  )

  # Header
  st.title("📧 Email Spam Detector")
  st.markdown("""
  This application uses Natural Language Processing (NLP) to detect spam emails.
  Enter your email text below to check if it's spam or not!
  """)

  # Sidebar
  st.sidebar.header("About")
  st.sidebar.info("""
  This spam detector uses:
  - TF-IDF Vectorization
  - Naive Bayes Classification
  - NLTK for text preprocessing
  """)

  # Main content
  col1, col2 = st.columns([2, 1])

  with col1:
      # Text input
      email_text = st.text_area(
          "Enter email text here:",
          height=200,
          placeholder="Type or paste your email content here..."
      )

      if st.button("Analyze Email", type="primary"):
          if email_text.strip() == "":
              st.warning("Please enter some text to analyze.")
          else:
              model, vectorizer = load_model()
              if model and vectorizer:
                  with st.spinner("Analyzing..."):
                      result = predict_spam(email_text, model, vectorizer)

                      # Display results
                      st.markdown("### Results")

                      # Create columns for results
                      result_col1, result_col2 = st.columns(2)

                      with result_col1:
                          if result['is_spam']:
                              st.error("🚨 Spam Detected!")
                          else:
                              st.success("✅ Not Spam")

                      with result_col2:
                          confidence_percentage = result['confidence'] * 100
                          st.metric(
                              label="Confidence Score",
                              value=f"{confidence_percentage:.2f}%"
                          )

  with col2:
      # Example emails
      st.markdown("### Example Emails")
      example_emails = {
          "Spam Example 1": "URGENT: You've won \$1,000,000! Click here to claim now!",
          "Spam Example 2": "FREE VIAGRA! Best prices guaranteed! Click now!",
          "Ham Example 1": "Hi John, can we reschedule our meeting to tomorrow at 2 PM?",
          "Ham Example 2": "Meeting minutes from yesterday's conference call attached."
      }

      for title, example in example_emails.items():
          if st.button(title):
              st.session_state['email_text'] = example
              st.experimental_rerun()

  # Display preprocessing info
  if st.checkbox("Show text preprocessing steps"):
      st.markdown("""
      ### Preprocessing Steps:
      1. Convert text to lowercase
      2. Remove special characters and numbers
      3. Remove extra whitespace
      4. Remove stopwords
      5. Apply lemmatization
      """)

if __name__ == "__main__":
  main()

# Created/Modified files during execution:
print("No files created/modified during execution")