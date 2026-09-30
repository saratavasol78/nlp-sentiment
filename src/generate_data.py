"""
تولید دیتاست مصنوعیِ نقد محصول (Product Review) برای تحلیل احساس متن

با ترکیب الگوهای جمله + واژگان مثبت/منفی/خنثی، جملات متنوع و واقع‌گرایانه
می‌سازیم تا کل پایپلاین NLP (پاک‌سازی متن، TF-IDF، LSTM) قابل اجرا باشد.

نکته برای استفاده‌ی واقعی در رزومه:
    برای یک نسخه‌ی رزومه‌پسندتر، این تابع را با یک دیتاست واقعی جایگزین
    کنید (مثل IMDB Reviews یا Amazon Reviews) — فقط کافیست یک CSV با
    ستون‌های text و label (negative/neutral/positive) در
    data/reviews.csv قرار دهید؛ بقیه‌ی پایپلاین بدون تغییر کار می‌کند.
"""

import numpy as np
import pandas as pd

from config import RAW_DATA_PATH, RANDOM_STATE

POSITIVE_TEMPLATES = [
    "I absolutely love this {product}, it works {adv_pos}.",
    "This {product} exceeded my expectations, {reason_pos}.",
    "Great {product}! {reason_pos} and I would buy it again.",
    "The {product} is fantastic, {reason_pos}.",
    "Highly recommend this {product}, {reason_pos}.",
    "Best {product} I've ever bought, {reason_pos}.",
]
NEGATIVE_TEMPLATES = [
    "I really dislike this {product}, it works {adv_neg}.",
    "This {product} was disappointing, {reason_neg}.",
    "Terrible {product}, {reason_neg} and I want a refund.",
    "The {product} is awful, {reason_neg}.",
    "Would not recommend this {product}, {reason_neg}.",
    "Worst {product} I've ever bought, {reason_neg}.",
]
NEUTRAL_TEMPLATES = [
    "The {product} is okay, nothing special.",
    "This {product} does what it says, {reason_neu}.",
    "An average {product}, {reason_neu}.",
    "The {product} arrived on time, {reason_neu}.",
    "It's a standard {product}, {reason_neu}.",
    "I have mixed feelings about this {product}, {reason_neu}.",
]

PRODUCTS = ["laptop", "phone", "headphones", "camera", "vacuum cleaner", "coffee maker",
            "backpack", "keyboard", "monitor", "watch"]
ADV_POS = ["perfectly", "wonderfully", "smoothly", "like a charm"]
ADV_NEG = ["poorly", "barely", "terribly", "inconsistently"]
REASON_POS = ["the build quality is excellent", "customer service was super helpful",
              "it's fast and reliable", "the battery life is amazing"]
REASON_NEG = ["it broke after a week", "customer service ignored me",
              "it's slow and unreliable", "the battery drains fast"]
REASON_NEU = ["but the price is a bit high", "though delivery took a while",
              "similar to other products in this range", "nothing stands out"]


def generate_reviews(n_per_class: int = 700) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []

    def fill(template):
        return template.format(
            product=rng.choice(PRODUCTS),
            adv_pos=rng.choice(ADV_POS),
            adv_neg=rng.choice(ADV_NEG),
            reason_pos=rng.choice(REASON_POS),
            reason_neg=rng.choice(REASON_NEG),
            reason_neu=rng.choice(REASON_NEU),
        )

    for _ in range(n_per_class):
        rows.append({"text": fill(rng.choice(POSITIVE_TEMPLATES)), "label": "positive"})
        rows.append({"text": fill(rng.choice(NEGATIVE_TEMPLATES)), "label": "negative"})
        rows.append({"text": fill(rng.choice(NEUTRAL_TEMPLATES)), "label": "neutral"})

    df = pd.DataFrame(rows)
    return df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


if __name__ == "__main__":
    df = generate_reviews()
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"دیتاست ساخته شد: {df.shape[0]} نمونه")
    print(df["label"].value_counts())
    print(f"ذخیره شد در: {RAW_DATA_PATH}")
