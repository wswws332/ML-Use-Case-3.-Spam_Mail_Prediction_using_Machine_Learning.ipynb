# ML-Use-Case-3.-Spam_Mail_Prediction_using_Machine_Learning.ipynb
# 📧 Spam Mail Prediction - ML Use Case 3

This project focuses on building a machine learning model to detect whether an email is spam or not based on its content. It demonstrates the use of natural language processing (NLP) and classification algorithms to solve a real-world binary classification problem.

---

## 📂 Dataset

**Source**: [Kaggle - Spam Mail Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

**Attributes**:
- **Label**: Indicates whether the message is `spam` or `ham` (not spam)
- **Message**: The text of the email/SMS message

---

## 🔍 Objective

To classify email messages as `spam` or `ham` based on their textual content using supervised machine learning techniques.

---

## 📊 Workflow Summary

1. **Data Cleaning & Preprocessing**:
   - Lowercasing text
   - Removing stopwords, punctuation, and special characters
   - Tokenization and stemming
   - Converting text into numerical features using:
     - Bag of Words (CountVectorizer)
     - TF-IDF Vectorizer

2. **Model Building**:
   - Naive Bayes (MultinomialNB)
   - Logistic Regression
   - Support Vector Machine (SVM)
   - Random Forest (optional)

3. **Model Evaluation**:
   - Accuracy, Precision, Recall, F1-score
   - Confusion Matrix
   - ROC-AUC curve

---

## 📈 Model Performance

| Model              | Accuracy | Precision | Recall | F1-Score |
|-------------------|----------|-----------|--------|----------|
| Naive Bayes       | 98%      | 0.97      | 0.95   | 0.96     |
| Logistic Regression | 96%    | 0.95      | 0.92   | 0.93     |
| SVM               | 97%      | 0.96      | 0.93   | 0.94     |

---

## 🧰 Requirements

```bash
Python 3.x
numpy
pandas
matplotlib
seaborn
scikit-learn
nltk
jupyter
```

Install using:
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

1. Clone the repository:
```bash
git clone https://github.com/your-username/ML-Use-case-3.-Spam_Mail_Prediction.git
cd ML-Use-case-3.-Spam_Mail_Prediction
```

2. Launch the notebook:
```bash
jupyter notebook Spam_Mail_Prediction_using_Machine_Learning.ipynb
```

---

## ✅ Future Work

- Add LSTM/Deep Learning-based text classifiers
- Deploy as a web app using Streamlit
- Create a browser plugin for real-time spam detection

---

## 🙋‍♀️ Author

**Aanchal Rai, SVNIT**  
[LinkedIn](https://www.linkedin.com/in/your-profile) | [GitHub](https://github.com/your-username)
## 📄 License

This project is licensed under the MIT License.
