import joblib
import re

model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

def predict_sentiment(text):
    
    def clean_text(t):
        t = t.lower()
        t = re.sub(r"<.*?>", " ", t)  # remove HTML tags
        t = re.sub(r"[^a-zA-Z]", " ", t)  # keep only letters
        return t
    
    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features).max()

    label = "Positive" if prediction == 1 else "Negative"
    return {"sentiment": label, "confidence": float(prob)}

sample = "This movie was absolutely fantastic! I loved it."
result = predict_sentiment(sample)
print(result)