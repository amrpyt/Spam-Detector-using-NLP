![image](https://github.com/user-attachments/assets/67123cc6-2c4d-40ba-9e21-fa2250b45146)

# **Spam Detector Using NLP**

This project demonstrates how to build an intelligent system to classify emails as **Spam** or **Not Spam** using Machine Learning and Natural Language Processing (NLP). It serves as a step-by-step guide for students to understand and implement such a system.

---

## **What Will You Learn?**

Through this project, you’ll gain hands-on experience in:

1. **Preprocessing textual data** using NLP techniques.
2. **Extracting meaningful numerical features** from text with TF-IDF Vectorization.
3. **Training a Machine Learning model** (Naive Bayes) for classification tasks.
4. **Building an interactive web-based application** using Streamlit.

---

## **How Does It Work?**

The system classifies an email message into two categories:

- **Spam**: Unwanted or promotional emails.
- **Not Spam**: Important, relevant emails.

This process involves:

1. **Text Preprocessing**: Cleaning and preparing text for analysis.
2. **Feature Extraction**: Converting text into numerical data using TF-IDF.
3. **Model Training**: Using a Naive Bayes classifier to detect patterns.
4. **Interactive Interface**: Analyzing new emails via a web app.

---

## **Project Workflow**

### **Dataset**

- **What’s the dataset?**
  - The **SMS Spam Collection Dataset** (labeled SMS messages).
  
- **How to get it?**
  - Download from [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).
  
- **File location:**
  - Save the file as `spam.csv` in the `data/` directory.

### **Text Preprocessing**

Before training the model, we clean the text:

1. **Lowercasing**: Converts characters to lowercase.
2. **Removing Special Characters**: Strips symbols, numbers, and extra spaces.
3. **Stopword Removal**: Removes common words like “is”, “the”, “and” using NLTK.  
   ![Stopword Removal](https://media.geeksforgeeks.org/wp-content/cdn-uploads/Stop-word-removal-using-NLTK.png)
4. **Lemmatization**: Reduces words to their base form (e.g., “running” → “run”).  
   ![Lemmatization](https://github.com/user-attachments/assets/71a3962f-d2f5-4a1d-b2eb-b47e233148ce)

**Why preprocessing?**

- It reduces noise.
- Ensures focus on meaningful words.

### **Feature Extraction with TF-IDF**

- **What is TF-IDF?**
  - A technique transforming text into numerical values based on:
    - **TF**: Term Frequency — how often a word appears.
    - **IDF**: Inverse Document Frequency — how unique a word is.
  
- **Why use it?**
  - It prioritizes relevant words over common ones.

- **Example**:
  - In the phrase "Win a free iPhone now!", words like "Win" and "free" get higher weights than "a" or "now".

### **Model Training**

- **Which model?**
  - **Naive Bayes Classifier** — fast, simple, and effective for text classification.

- **Why Naive Bayes?**
  - Works well with text.
  - Calculates probabilities for each class.

### **Interactive Interface with Streamlit**

- **What is Streamlit?**
  - A Python library for creating interactive web apps.

- **What does it do?**
  - Lets users input an email and see if it’s Spam or Not Spam, with a confidence score.

---

## **Directory Structure**

```
spam_detector/
├── data/
│   └── spam.csv          # Dataset
├── model/
│   ├── spam_model.joblib # Trained model
│   └── vectorizer.joblib # TF-IDF vectorizer
├── src/
│   ├── train_model.py    # Script to train the model
│   └── predict.py        # Script to make predictions
├── GUI/
│   └── main.py           # Streamlit app
├── requirements.txt      # Required libraries
└── README.md             # Project documentation
```

---

## **Step-by-Step Setup**

### **Clone the Repository**

```bash
git clone https://github.com/your_username/spam-detector-nlp.git
cd spam-detector-nlp
```

### **Create a Virtual Environment**

```bash
python -m venv .venv
```

### **Activate the Virtual Environment**

- **Windows:**
  ```bash
  .venv\Scripts\activate
  ```
- **macOS/Linux:**
  ```bash
  source .venv/bin/activate
  ```

### **Install Dependencies**

```bash
pip install -r requirements.txt
```

---

## **Usage Instructions**

### **Train the Model**

Run the training script:

```bash
python src/train_model.py
```

This generates:

- `spam_model.joblib` (trained model).
- `vectorizer.joblib` (TF-IDF vectorizer).

### **Run the Streamlit App**

Start the web app:

```bash
streamlit run GUI/main.py
```

### **Analyze Emails**

- Open the link (e.g., [http://localhost:8501](http://localhost:8501)).
- Input an email and click **Analyze Email**.

---

## **Example Emails**

1. **Spam Email:**
   ```
   Congratulations! You’ve won $1,000,000! Click here to claim now!
   ```
   - **Prediction:** Spam
   - **Confidence Score:** 95%

2. **Not Spam Email:**
   ```
   Hi John, can we reschedule our meeting to tomorrow at 2 PM?
   ```
   - **Prediction:** Not Spam
   - **Confidence Score:** 99%

---

## **Key Technologies**

1. **Python**: The programming language.
2. **Libraries**:
   - **Streamlit**: For the web interface.
   - **Scikit-learn**: For the ML model.
   - **NLTK**: For text preprocessing.
   - **Joblib**: For saving/loading models.

---

## **How the System Works**

1. **Preprocessing**: Clean and standardize the input text.
2. **Feature Extraction**: Convert text into numerical data.
3. **Training**: Train the Naive Bayes model.
4. **Prediction**: Classify new emails.

---

## **Future Enhancements**

1. **Support for Additional Languages**.
2. **Advanced Models**: Experiment with deep learning models.
3. **Batch Classification**: Process multiple emails simultaneously.

---

## **Learning Outcomes**

- End-to-end understanding of text classification.
- Experience with data preprocessing and feature extraction.
- Deployment skills with Streamlit.

---

## **Credits**

Made with ❤️ by **Amr Alkhouli**

---

## **License**

This project is licensed under the [MIT License](LICENSE).
