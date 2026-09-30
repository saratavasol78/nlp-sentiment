# Text Sentiment Analysis — NLP

طبقه‌بندی احساس متن (مثبت/خنثی/منفی) با دو رویکرد: یک baseline سریع
(TF-IDF + Logistic Regression) و یک مدل عمیق (Embedding + BiLSTM آموزش‌دیده
از صفر با TensorFlow)، به‌همراه API آماده‌ی سرویس‌دهی.

## ساختار پروژه
```
04-nlp-sentiment/
├── data/                        # دیتاست نقد محصول (تولید می‌شود)
├── models/                      # مدل‌های ذخیره‌شده
├── src/
│   ├── config.py
│   ├── generate_data.py         # دیتاست مصنوعی نقد محصول
│   ├── preprocess.py            # پاک‌سازی متن
│   ├── train_tfidf_baseline.py  # baseline: TF-IDF + Logistic Regression
│   ├── train_lstm.py            # مدل عمیق: Embedding + BiLSTM
│   └── api.py                   # سرویس‌دهی با FastAPI
├── requirements.txt
└── README.md
```

## نصب و اجرا
```bash
pip install -r requirements.txt

cd src
python generate_data.py            # ساخت دیتاست
python train_tfidf_baseline.py     # baseline سریع
python train_lstm.py               # مدل عمیق
uvicorn api:app --reload           # اجرای API
```

## درباره‌ی دیتاست و مسیر ارتقا
دیتاست پیش‌فرض مصنوعی و مبتنی‌بر الگوهای جمله است تا کل پایپلاین بدون
دانلود قابل اجراست باشد. برای رزومه‌ی واقعی:
- دیتاست IMDB یا Amazon Reviews را با ستون‌های `text` و `label` در
  `data/reviews.csv` جایگزین کنید — بقیه‌ی کد بدون تغییر کار می‌کند.
- برای دقت بالاتر، می‌توانید `train_lstm.py` را با فاین‌تیون یک مدل
  transformer از پیش‌آموزش‌دیده (مثل DistilBERT با کتابخانه‌ی
  `transformers`) جایگزین کنید.

## بولت‌های پیشنهادی برای رزومه
- Built an NLP sentiment-classification pipeline comparing a TF-IDF +
  Logistic Regression baseline against a BiLSTM deep learning model
  trained from scratch with TensorFlow.
- Implemented a full text preprocessing and tokenization pipeline and
  deployed the trained model as a REST API for real-time sentiment scoring.
