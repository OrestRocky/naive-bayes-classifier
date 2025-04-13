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
