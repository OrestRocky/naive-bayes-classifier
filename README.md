 🧠 Naive Bayes Classifier

A professional-level implementation of a Naive Bayes classifier in Python for spam detection. This project not only demonstrates the practical application of the algorithm, but also explains the theory behind it — making it a foundational example of probabilistic reasoning in AI systems.



Overview

Naive Bayes is a probabilistic classifier based on **Bayes' Theorem**, widely used in machine learning for tasks like text classification, email filtering, and recommendation systems. It's called "naive" because it assumes **independence** between features — an assumption that simplifies computation while remaining effective in practice.



 Mathematical Foundation

Bayes' Theorem:

\[
P(C|X) = \frac{P(X|C) \cdot P(C)}{P(X)}
\]

Where:
- \( P(C|X) \) — Posterior probability of class \( C \) given input \( X \)
- \( P(X|C) \) — Likelihood of data \( X \) given class \( C \)
- \( P(C) \) — Prior probability of class \( C \)
- \( P(X) \) — Marginal likelihood of input \( X \)

In **Naive Bayes**, we assume that features \( x_1, x_2, ..., x_n \) are conditionally independent:

\[
P(X|C) = \prod_{i=1}^{n} P(x_i|C)
\]

This simplifies calculations and speeds up training, making it suitable for real-time systems.


Project Structure

```
naive-bayes-classifier/
├── src/                  # Source code (classifier class)
│   └── classifier.py
├── data/                 # Dataset (CSV with spam/ham messages)
│   └── spam_emails.csv
├── notebook/             # Jupyter Notebook for demonstration
│   └── naive_bayes_demo.ipynb
├── requirements.txt      # Python dependencies
└── README.md             # This file
```



Core Implementation (Python)

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

class NaiveBayesTextClassifier:
    def __init__(self):
        self.model = Pipeline([
            ('vectorizer', CountVectorizer()),
            ('classifier', MultinomialNB())
        ])

    def train(self, texts, labels):
        self.model.fit(texts, labels)

    def predict(self, texts):
        return self.model.predict(texts)

    def score(self, texts, labels):
        return self.model.score(texts, labels)
```
 Demo Notebook Features

The `notebook/naive_bayes_demo.ipynb` file includes:
- Data loading with pandas
- Training and prediction
- Accuracy evaluation
- Sample classification of new messages

 Example Output

```python
# Sample predictions:
predictions = model.predict([
    "Claim your free prize now!",
    "Hey, how are you doing today?",
    "Let's meet at the cafe."
])
print(predictions)
# Output: ['spam' 'ham' 'ham']

# Accuracy of the model:
accuracy = model.score(texts, labels)
print(f"Model accuracy: {accuracy:.2f}")
# Output: Model accuracy: 0.83
```

How to Run

```bash
git clone https://github.com/yourusername/naive-bayes-classifier.git
cd naive-bayes-classifier
pip install -r requirements.txt
jupyter notebook notebook/naive_bayes_demo.ipynb
```

 Why This Matters in AI

Naive Bayes illustrates core ideas behind many AI systems:
- **Probabilistic reasoning**
- **Uncertainty modeling**
- **Lightweight decision-making**

It’s foundational in building explainable, efficient systems, especially in NLP and decision support.



 Future Ideas
- Confusion matrix visualization
- Advanced preprocessing (n-grams, lemmatization)
- Comparison with Logistic Regression and Decision Trees


## 📄 License
MIT License

