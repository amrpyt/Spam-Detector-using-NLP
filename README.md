![image](https://github.com/user-attachments/assets/c1f1c827-49de-4476-adb2-f60da83d2dcb)![image](https://github.com/user-attachments/assets/67123cc6-2c4d-40ba-9e21-fa2250b45146)
# Spam Detector Using NLP

This project demonstrates how to build an intelligent system to classify emails as **“Spam”** or **“Not Spam”** using Machine Learning and Natural Language Processing (NLP). It’s structured as a comprehensive guide for students to understand and implement such a system step-by-step.

## What Will You Learn?

Through this project, you’ll gain hands-on experience in:

1. **Preprocessing textual data** using NLP techniques.
2. **Extracting meaningful numerical features** from text with TF-IDF Vectorization.
3. **Training a Machine Learning model** (Naive Bayes) for classification tasks.
4. **Building an interactive web-based application** using Streamlit.

## How Does It Work?

The system processes an email message and classifies it into two categories:

- **Spam**: Unwanted or promotional emails, often irrelevant.
- **Not Spam**: Important and meaningful emails.

This process involves:

1. **Text Preprocessing**: Cleaning and preparing the text for analysis.
2. **Feature Extraction**: Converting the text into numerical data using TF-IDF.
3. **Model Training**: Using a Naive Bayes classifier to learn patterns in the data.
4. **Interactive Interface**: Allowing users to input and analyze new emails.

## Project Workflow

### Dataset

- **What’s the dataset?**
  - We use the **SMS Spam Collection Dataset**, which consists of labeled SMS messages (Spam/Not Spam).
- **How to get it?**
  - Download the dataset from [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).
- **File location:**
  - Save the file as `spam.csv` in the `data/` directory.

### Text Preprocessing

Before training the model, we need to clean the raw text. This step ensures that the text is in a consistent and machine-readable format:

1. **Lowercasing**: Convert all characters to lowercase for uniformity.


2. **Removing Special Characters**: Eliminate symbols, numbers, and extra spaces.


3. **Stopword Removal**: Remove common words like “is”, “the”, “and” using libraries like NLTK. ![image](https://media.geeksforgeeks.org/wp-content/cdn-uploads/Stop-word-removal-using-NLTK.png)


4. **Lemmatization**: Reduce words to their base form (e.g., “running” → “run”).
   ![image](https://github.com/user-attachments/assets/71a3962f-d2f5-4a1d-b2eb-b47e233148ce)


**Why is preprocessing important?**

- It reduces noise in the data.
- It ensures the model focuses on meaningful words.

### Feature Extraction with TF-IDF

- **What is TF-IDF?**
  - A technique to transform text into numerical values by calculating:
    - **TF (Term Frequency)**: How often a word appears in a message.
    - **IDF (Inverse Document Frequency)**: How unique a word is across all messages.
- **Why use it?**
  - To assign higher importance to relevant words and ignore generic ones.
- **Example:**
  - In the text “Win a free iPhone now!”, words like “Win” and “free” will have higher weights than “a” or “now”.

### Model Training

- **Which model are we using?**
  - **Naive Bayes Classifier**, a simple and effective algorithm for text classification tasks.
- **Why Naive Bayes?**
  - It works well with text data.
  - It’s fast and easy to implement.
- **What does it do?**
  - It calculates probabilities for each class (Spam or Not Spam) and selects the class with the highest probability.

### Interactive Interface with Streamlit

- **What is Streamlit?**
  - A Python library for creating simple and interactive web applications.
- **What does the app do?**
  - Allows users to input an email, analyze it, and see if it’s Spam or Not Spam, along with a confidence score.

## Directory Structure

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

## Step-by-Step Setup

### Clone the Repository

```bash
git clone https://github.com/your_username/spam-detector-nlp.git
cd spam-detector-nlp
```

### Create a Virtual Environment

```bash
python -m venv .venv
```

### Activate the Virtual Environment

- **Windows:**
  ```bash
  .venv\Scripts\activate
  ```
- **macOS/Linux:**
  ```bash
  source .venv/bin/activate
  ```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage Instructions

### Train the Model

Run the training script to process the data and train the Naive Bayes model:

```bash
python src/train_model.py
```

This will generate:

- `spam_model.joblib`: The trained model.
- `vectorizer.joblib`: The TF-IDF vectorizer.

### Run the Streamlit App

Start the interactive web app:

```bash
streamlit run GUI/main.py
```

### Analyze Emails

- Open the provided link in your terminal (e.g., [http://localhost:8501](http://localhost:8501)).
- Input an email message into the app and click **Analyze Email**.

### Example Emails

Here are some sample emails to test:

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

## Key Technologies

1. **Python:** The programming language used for the project.
2. **Libraries:**
   - **Streamlit:** For the web-based interface.
   - **Scikit-learn:** For building and evaluating the ML model.
   - **NLTK:** For preprocessing text.
   - **Joblib:** For saving and loading the trained model.

## How the System Works

1. **Preprocessing:**
   - Clean and standardize the input text.
2. **Feature Extraction:**
   - Convert the text into numerical data using TF-IDF.
3. **Training:**
   - Train the Naive Bayes model on labeled data.
4. **Prediction:**
   - Use the trained model to classify new email messages.

## Future Enhancements

1. **Support for Additional Languages:**
   - Extend the system to handle non-English emails.
2. **Advanced Models:**
   - Experiment with deep learning models like RNNs or Transformers for better accuracy.
3. **Batch Classification:**
   - Enable processing multiple emails at once.

## Learning Outcomes

- Understand the end-to-end pipeline for text classification.
- Learn how to preprocess, transform, and analyze textual data.
- Gain experience in deploying a Machine Learning model using Streamlit.

This version is tailored to be both instructive and comprehensive, ideal for students or anyone learning NLP and Machine Learning concepts.

## Credits

Made with ❤️ by **Amr Alkhouli**

## License

This project is licensed under the [MIT License](LICENSE).
```
