# **Spam Detector using NLP**
An intelligent system to classify email messages as **Spam** or **Not Spam**, leveraging **Machine Learning** and **Natural Language Processing (NLP)**.

---

## **Features**
- **Email Classification**: Detect whether an email is spam or not.
- **Machine Learning**: Uses **Naive Bayes** for training and classification.
- **NLP Preprocessing**: Includes tokenization, stopword removal, lemmatization, and TF-IDF vectorization.
- **Interactive Interface**: Powered by **Streamlit** for user-friendly email analysis.
- **Example Emails**: Preloaded examples to demonstrate functionality.

---

## **Project Structure**
```
spam_detector/
├── data/
│   └── spam.csv                # Dataset for training the model
├── model/
│   └── spam_model.joblib       # Trained model
│   └── vectorizer.joblib       # TF-IDF vectorizer
├── src/
│   ├── train_model.py          # Script to train the model
│   └── predict.py              # Script to make predictions
├── GUI/
│   └── main.py                 # Streamlit app
├── requirements.txt            # Required Python libraries
└── README.md                   # Project documentation
```

---

## **Setup and Installation**

### **1. Clone the Repository**
```bash
git clone https://github.com/your_username/spam-detector-nlp.git
cd spam-detector-nlp
```

### **2. Create a Virtual Environment**
```bash
python -m venv .venv
```

### **3. Activate the Virtual Environment**
- **Windows**:
  ```bash
  .venv\Scripts\activate
  ```
- **macOS/Linux**:
  ```bash
  source .venv/bin/activate
  ```

### **4. Install Dependencies**
```bash
pip install -r requirements.txt
```

---

## **Usage**

### **1. Train the Model**
To train the model, use the `train_model.py` script:
```bash
python src/train_model.py
```
This will generate the `spam_model.joblib` and `vectorizer.joblib` files in the `model/` directory.

### **2. Run the Streamlit App**
Start the app with the following command:
```bash
streamlit run GUI/main.py
```

### **3. Open the App**
Open your browser and navigate to the URL provided in the terminal (e.g., `http://localhost:8501`).

### **4. Test the App**
- Paste an email text into the input box.
- Click "Analyze Email" to classify the email as **Spam** or **Not Spam**.
- Review the confidence score of the prediction.

---

## **Example Emails**
Here are some sample emails you can test:
1. **Spam Email**:
   > URGENT: You've won $1,000,000! Click here to claim now!
2. **Not Spam Email**:
   > Hi John, can we reschedule our meeting to tomorrow at 2 PM?

---

## **Technologies Used**
- **Programming Language**: Python
- **Libraries**:
  - **Streamlit**: For the web app interface
  - **Scikit-learn**: For machine learning
  - **NLTK**: For text preprocessing
  - **Joblib**: For model serialization

---

## **How It Works**
1. **Preprocessing**:
   - Convert text to lowercase.
   - Remove special characters, numbers, and extra whitespace.
   - Remove stopwords using NLTK.
   - Apply lemmatization to standardize words.

2. **Feature Extraction**:
   - Use **TF-IDF Vectorizer** to transform text into numerical features.

3. **Model Training**:
   - Train a **Naive Bayes Classifier** using labeled data.

4. **Prediction**:
   - Classify new email text and provide confidence scores.

---

## **Contributing**
We welcome contributions to enhance this project. Feel free to:
- Fork the repository.
- Create feature branches.
- Submit pull requests.

---

## **Future Enhancements**
- Add support for additional languages.
- Implement deep learning models like **RNNs** or **Transformers** for better accuracy.
- Provide a batch email classification feature.

---

## **Credits**
- **Made with ❤️ by [Amr Alkhouli](https://github.com/your_username)**

---

## **License**
This project is licensed under the [MIT License](LICENSE).
