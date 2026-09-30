"""
مدل عمیق: Embedding + LSTM (آموزش از صفر با TensorFlow/Keras)

برخلاف مدل‌های transformer از پیش‌آموزش‌دیده (که نیاز به دانلود وزن‌های
سنگین از هاگینگ‌فیس دارن)، این مدل کاملاً از صفر روی داده‌ی خودمون آموزش
می‌بینه و برای دیتاست‌های کوچک/متوسط سریع و کاربردیه.

اجرا:
    python src/train_lstm.py

نکته برای ارتقا به سطح production:
    برای دقت بالاتر روی داده‌ی واقعی، می‌تونید embedding از صفر رو با
    GloVe/Word2Vec از پیش‌آموزش‌دیده جایگزین کنید، یا مدل رو با یک
    transformer کوچک (DistilBERT) از کتابخونه‌ی transformers فاین‌تیون
    کنید (نیاز به اینترنت برای دانلود وزن‌ها دارد).
"""

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
from tensorflow.keras import layers

from config import (
    BATCH_SIZE,
    EPOCHS,
    LSTM_MODEL_PATH,
    MAX_SEQUENCE_LENGTH,
    MAX_VOCAB_SIZE,
    MODELS_DIR,
    RANDOM_STATE,
    RAW_DATA_PATH,
    SENTIMENT_CLASSES,
    TOKENIZER_PATH,
)
from preprocess import clean_text


def build_model(vocab_size: int, num_classes: int) -> keras.Model:
    inputs = keras.Input(shape=(MAX_SEQUENCE_LENGTH,))
    x = layers.Embedding(vocab_size, 64, mask_zero=True)(inputs)
    x = layers.Bidirectional(layers.LSTM(64, dropout=0.3, recurrent_dropout=0.0))(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="sentiment_lstm")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RAW_DATA_PATH)
    df["text_clean"] = df["text"].apply(clean_text)

    label_encoder = LabelEncoder()
    label_encoder.fit(SENTIMENT_CLASSES)
    y = label_encoder.transform(df["label"])

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["text_clean"], y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    tokenizer = keras.preprocessing.text.Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train_text)

    def to_padded_sequences(texts):
        seqs = tokenizer.texts_to_sequences(texts)
        return keras.preprocessing.sequence.pad_sequences(
            seqs, maxlen=MAX_SEQUENCE_LENGTH, padding="post", truncating="post"
        )

    X_train = to_padded_sequences(X_train_text)
    X_test = to_padded_sequences(X_test_text)

    vocab_size = min(MAX_VOCAB_SIZE, len(tokenizer.word_index) + 1)
    model = build_model(vocab_size, len(SENTIMENT_CLASSES))
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
    ]

    model.fit(
        X_train,
        y_train,
        validation_split=0.15,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=2,
    )

    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nدقت روی مجموعه‌ی تست: {test_acc:.4f}")

    model.save(LSTM_MODEL_PATH)
    joblib.dump({"tokenizer": tokenizer, "label_encoder": label_encoder}, TOKENIZER_PATH)
    print(f"مدل ذخیره شد در: {LSTM_MODEL_PATH}")
    print(f"توکنایزر ذخیره شد در: {TOKENIZER_PATH}")


if __name__ == "__main__":
    main()
