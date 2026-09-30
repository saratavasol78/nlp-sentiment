"""
سرویس‌دهی مدل تحلیل احساس متن با FastAPI (از مدل LSTM استفاده می‌کند)

اجرا:
    uvicorn src.api:app --reload --port 8000
"""

import joblib
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow import keras

from config import LSTM_MODEL_PATH, MAX_SEQUENCE_LENGTH, TOKENIZER_PATH
from preprocess import clean_text

app = FastAPI(
    title="Text Sentiment Analysis API",
    description="طبقه‌بندی احساس متن (positive / neutral / negative)",
    version="1.0.0",
)

model = tf.keras.models.load_model(LSTM_MODEL_PATH)
bundle = joblib.load(TOKENIZER_PATH)
tokenizer = bundle["tokenizer"]
label_encoder = bundle["label_encoder"]


class TextRequest(BaseModel):
    text: str


class SentimentResponse(BaseModel):
    label: str
    probabilities: dict[str, float]


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Sentiment API is running"}


@app.post("/predict", response_model=SentimentResponse)
def predict(request: TextRequest):
    cleaned = clean_text(request.text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = keras.preprocessing.sequence.pad_sequences(
        seq, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post"
    )
    probs = model.predict(padded, verbose=0)[0]
    pred_idx = int(probs.argmax())
    label = label_encoder.inverse_transform([pred_idx])[0]

    return SentimentResponse(
        label=label,
        probabilities={
            cls: float(p) for cls, p in zip(label_encoder.classes_, probs)
        },
    )
