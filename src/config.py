"""تنظیمات مرکزی پروژه: مسیرها و هایپرپارامترها."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

RAW_DATA_PATH = DATA_DIR / "reviews.csv"
TFIDF_MODEL_PATH = MODELS_DIR / "tfidf_logreg.joblib"
LSTM_MODEL_PATH = MODELS_DIR / "lstm_sentiment.keras"
TOKENIZER_PATH = MODELS_DIR / "tokenizer.joblib"

SENTIMENT_CLASSES = ["negative", "neutral", "positive"]
MAX_VOCAB_SIZE = 5000
MAX_SEQUENCE_LENGTH = 40
RANDOM_STATE = 42
EPOCHS = 15
BATCH_SIZE = 32
