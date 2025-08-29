from fastapi import FastAPI
import joblib
import re

model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

app = FastAPI(title="Sentiment Analysis API")

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    return text

@app.get("/")
def home():
    return {"message": "Sentiment analysis API is working"}

@app.post("/predict")
def predict(text: str):
    cleaned = clean_text(text)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]
    prob = model.predict_proba(features).max()

    label = "Positive" if prediction == 1 else "Negative"
    return {"sentiment": label, "confidence": float(prob)}

# curl -X 'POST' \
#  'http://localhost:8000/predict?text=Hello%20World' \
#  -H 'accept: application/json' \
#  -d ''
# > uvicorn app:app --reload