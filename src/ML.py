import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


SPLIT = "valid"  # Change to "test" only for final evaluation.

if SPLIT not in {"valid", "test"}:
    raise ValueError("SPLIT must be 'valid' or 'test'.")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "dataset"
OUTPUT_DIR = PROJECT_ROOT / "Outputs"
TRAIN_PATH = DATA_DIR / "UIT-VSFC_train.csv"
EVAL_PATH = DATA_DIR / f"UIT-VSFC_{SPLIT}.csv"
LABEL_NAMES = ["negative", "neutral", "positive"]


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


train = pd.read_csv(TRAIN_PATH).dropna(subset=["text", "label"]).copy()
evaluation = (
    pd.read_csv(EVAL_PATH)
    .dropna(subset=["text", "label"])
    .reset_index(drop=True)
)
if SPLIT == "valid":
    evaluation["sample_id"] = evaluation.index
    evaluation["raw_text"] = evaluation["text"].astype(str)

train["text"] = train["text"].apply(clean_text)
evaluation["text"] = evaluation["text"].apply(clean_text)


X_train = train["text"]
y_train = train["label"]
X_eval = evaluation["text"]
y_eval = evaluation["label"]

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_eval_tfidf = vectorizer.transform(X_eval)

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
)
model.fit(X_train_tfidf, y_train)
predictions = model.predict(X_eval_tfidf)

print(f"Evaluation split: {SPLIT.upper()}")
print(f"Accuracy: {accuracy_score(y_eval, predictions):.4f}")
print(f"Macro F1: {f1_score(y_eval, predictions, average='macro'):.4f}")
print(f"Weighted F1: {f1_score(y_eval, predictions, average='weighted'):.4f}")
print("\nClassification report:")
print(classification_report(y_eval, predictions))

cm = confusion_matrix(y_eval, predictions, labels=LABEL_NAMES)
display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_NAMES)
display.plot()
plt.title(f"Confusion Matrix - {SPLIT.upper()}")
plt.tight_layout()
plt.show()

if SPLIT == "valid":
    results_df = pd.DataFrame(
        {
            "sample_id": evaluation["sample_id"],
            "text": evaluation["raw_text"],
            "true_label": y_eval,
            "predicted_label": predictions,
        }
    )
    results_df["is_correct"] = (
        results_df["true_label"] == results_df["predicted_label"]
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    prediction_path = OUTPUT_DIR / "ml_valid_predictions.csv"
    results_df.to_csv(prediction_path, index=False, encoding="utf-8-sig")
    print(f"Validation predictions saved to: {prediction_path}")

    errors_df = results_df[~results_df["is_correct"]].copy()
    errors_df["neutral_first"] = errors_df["true_label"].ne("neutral")
    errors_df = errors_df.sort_values("neutral_first", kind="stable").drop(
        columns="neutral_first"
    )

    print("\nError analysis (neutral errors first):")
    with pd.option_context("display.max_colwidth", None):
        print(errors_df.head(20).to_string(index=False))
else:
    print(
        "Test split is reserved for final evaluation; "
        "detailed error analysis is skipped."
    )
