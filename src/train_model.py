import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import joblib
import os

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

def preprocess_text(text):
  # Convert to lowercase
  text = text.lower()

  # Remove special characters and numbers
  text = re.sub(r'[^a-zA-Z\s]', '', text)

  # Remove extra whitespace
  text = re.sub(r'\s+', ' ', text).strip()

  # Tokenization and removing stopwords
  stop_words = set(stopwords.words('english'))
  lemmatizer = WordNetLemmatizer()

  words = text.split()
  words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]

  return ' '.join(words)

def train_spam_model():
  # Create directories if they don't exist
  os.makedirs('data', exist_ok=True)
  os.makedirs('model', exist_ok=True)

  # Load the spam dataset (you'll need to download this)
  # Sample dataset: https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
  df = pd.read_csv('data/spam.csv', encoding='latin-1')
  df = df.rename(columns={'v1': 'label', 'v2': 'text'})

  # Preprocess the text
  df['processed_text'] = df['text'].apply(preprocess_text)

  # Convert labels to binary
  df['label'] = df['label'].map({'ham': 0, 'spam': 1})

  # Split the data
  X_train, X_test, y_train, y_test = train_test_split(
      df['processed_text'], 
      df['label'], 
      test_size=0.2, 
      random_state=42
  )

  # Create TF-IDF vectors
  vectorizer = TfidfVectorizer(max_features=5000)
  X_train_tfidf = vectorizer.fit_transform(X_train)
  X_test_tfidf = vectorizer.transform(X_test)

  # Train the model
  model = MultinomialNB()
  model.fit(X_train_tfidf, y_train)

  # Evaluate the model
  y_pred = model.predict(X_test_tfidf)
  print("\nClassification Report:")
  print(classification_report(y_test, y_pred))

  # Save the model and vectorizer
  joblib.dump(model, 'model/spam_model.pkl')
  joblib.dump(vectorizer, 'model/vectorizer.pkl')

  print("\nModel and vectorizer saved successfully!")

if __name__ == "__main__":
  train_spam_model()

# Created/Modified files during execution:
print("Created/Modified files:")
for file in ["model/spam_model.pkl", "model/vectorizer.pkl"]:
  print(file)