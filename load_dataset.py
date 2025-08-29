from datasets import load_dataset

import pandas as pd

import re
from sklearn.feature_extraction.text import TfidfVectorizer

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