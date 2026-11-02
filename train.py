"""
Fake News Detection: TF-IDF + LinearSVC
Dataset: "Fake and Real News Dataset" (ISOT) from Kaggle
         put Fake.csv and True.csv inside the data/ folder
Run:     python train.py
"""
import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

DATA_DIR = Path("data")
MODEL_PATH = Path("model.joblib")
RANDOM_STATE = 42


def clean_text(text: str) -> str:
    text = str(text)
    # Real articles start with "CITY (Reuters) - ", which leaks the label.
    # Remove it so the model learns writing patterns, not the source tag.
    text = re.sub(r"^.{0,80}?\(Reuters\)\s*-\s*", "", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)   # urls
    text = re.sub(r"[^a-zA-Z\s]", " ", text)        # keep letters only
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def load_data() -> pd.DataFrame:
    fake = pd.read_csv(DATA_DIR / "Fake.csv")
    true = pd.read_csv(DATA_DIR / "True.csv")
    fake["label"] = 0   # 0 = fake
    true["label"] = 1   # 1 = real
    df = pd.concat([fake, true], ignore_index=True)

    # use title + text only (subject/date would leak the label)
    df["content"] = (df["title"].fillna("") + " " + df["text"].fillna("")).map(clean_text)

    before = len(df)
    df = df[df["content"].str.len() > 20]
    df = df.drop_duplicates(subset="content")
    print(f"Rows: {before} -> {len(df)} after removing empty/duplicate articles")
    print(df["label"].value_counts().rename({0: "fake", 1: "real"}), "\n")
    return df[["content", "label"]]


def main():
    df = load_data()

    # split BEFORE fitting TF-IDF so test data never influences the vectorizer
    X_train, X_test, y_train, y_test = train_test_split(
        df["content"], df["label"],
        test_size=0.2, stratify=df["label"], random_state=RANDOM_STATE,
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                  max_df=0.7, min_df=3, sublinear_tf=True)),
        ("clf", LinearSVC(C=1.0, random_state=RANDOM_STATE)),
    ])
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    print(f"Accuracy: {accuracy_score(y_test, pred):.4f}\n")
    print(classification_report(y_test, pred, target_names=["fake", "real"], digits=4))

    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=["fake", "real"], cmap="Blues")
    plt.title("Fake News Detection - Confusion Matrix")
    plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
