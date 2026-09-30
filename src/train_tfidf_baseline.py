"""
مدل پایه: TF-IDF + Logistic Regression

یک baseline سریع و قوی که همیشه قبل از رفتن سراغ مدل‌های پیچیده‌تر (LSTM،
Transformer) باید ساخته بشه — نشون‌دادن این عادت خوب مهندسی هم رزومه‌پسنده.

اجرا:
    python src/generate_data.py   # یک‌بار
    python src/train_tfidf_baseline.py
"""

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from config import MODELS_DIR, RANDOM_STATE, RAW_DATA_PATH, TFIDF_MODEL_PATH
from preprocess import clean_text


def main():
    df = pd.read_csv(RAW_DATA_PATH)
    df["text_clean"] = df["text"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["text_clean"], df["label"], test_size=0.2, stratify=df["label"],
        random_state=RANDOM_STATE,
    )

    pipe = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]
    )
    pipe.fit(X_train, y_train)

    preds = pipe.predict(X_test)
    print(classification_report(y_test, preds))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, TFIDF_MODEL_PATH)
    print(f"مدل ذخیره شد در: {TFIDF_MODEL_PATH}")


if __name__ == "__main__":
    main()
