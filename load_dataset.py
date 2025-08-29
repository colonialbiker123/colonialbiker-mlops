from datasets import load_dataset

import pandas as pd

import re
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

dataset = load_dataset("imdb")

# print(dataset)
# print(dataset["train"][0])

train_df = pd.DataFrame(dataset["train"])
test_df = pd.DataFrame(dataset["test"])

# print(train_df.tail())

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text) # Remove HTML Tags if any
    text = re.sub(r"[^a-zA-Z]", " ", text) # Keep only alphabets
    return text

train_df["clean_text"] = train_df["text"].apply(clean_text)
test_df["clean_text"] = test_df["text"].apply(clean_text)

vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train_df["clean_text"])
X_test = vectorizer.fit_transform(test_df["clean_text"])

y_train = train_df["label"]
y_test = test_df["label"]

# print("Train shape: ", X_train.shape)
# print("Test Shape: ", X_test.shape)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
# print("Accuracy: ", acc)
# print("Classification Report: ", classification_report(y_test, y_pred))
# print("Confusion matrix: ", confusion_matrix(y_test, y_pred))

joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
# print("Model and vectorizer saved successfully")