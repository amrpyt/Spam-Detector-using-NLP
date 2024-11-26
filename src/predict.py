import joblib
from train_model import preprocess_text


def load_model():
    model = joblib.load("model/spam_model.joblib")
    vectorizer = joblib.load("model/vectorizer.joblib")
    return model, vectorizer


def predict_spam(text, model, vectorizer):
    # Preprocess the input text
    processed_text = preprocess_text(text)

    # Transform the text using the vectorizer
    text_vector = vectorizer.transform([processed_text])

    # Make prediction
    prediction = model.predict(text_vector)[0]
    probability = model.predict_proba(text_vector)[0]

    return {
        "is_spam": bool(prediction),
        "confidence": float(probability[1]) if prediction else float(probability[0]),
    }


def main():
    # Load the model
    model, vectorizer = load_model()

    # Example usage
    while True:
        text = input("\nEnter an email text (or 'quit' to exit): ")
        if text.lower() == "quit":
            break

        result = predict_spam(text, model, vectorizer)
        print(f"\nResult: {'SPAM' if result['is_spam'] else 'NOT SPAM'}")
        print(f"Confidence: {result['confidence']:.2%}")


if __name__ == "__main__":
    main()
