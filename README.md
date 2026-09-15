# Vietnamese Student Feedback Sentiment Classification

Sentiment classification on UIT-VSFC using a TF-IDF baseline and an LSTM with Vietnamese word segmentation and pretrained Word2Vec.

## Overview

This project classifies Vietnamese student feedback as **negative**, **neutral**, or **positive**. It compares a compact classical baseline with a neural sequence model while keeping the official training, validation, and test sets separate.

The repository follows a reproducible workflow: training data fits the models, validation data supports development and sample-level analysis, and test data is used once for aggregate final evaluation.

## Dataset

The project uses the three official UIT-VSFC CSV splits in `dataset/`:

| Split | Negative | Neutral | Positive | Total |
|---|---:|---:|---:|---:|
| Train | 5,325 | 458 | 5,643 | 11,426 |
| Validation | 705 | 73 | 805 | 1,583 |
| Test | 1,409 | 167 | 1,590 | 3,166 |

Each row contains Vietnamese feedback text and one sentiment label. Neutral is the smallest class in every split.

## Project Structure

```text
Vietnamese Student Feedback Classification/
├── dataset/
│   ├── UIT-VSFC_train.csv
│   ├── UIT-VSFC_valid.csv
│   └── UIT-VSFC_test.csv
├── embeddings/                    # External Word2Vec file; ignored by Git
├── models/
│   └── lstm_sentiment.pt
├── Notebooks/
│   └── 01_eda.ipynb
├── Outputs/
│   ├── ml_valid_predictions.csv
│   ├── dl_valid_predictions.csv
│   └── compare_valid_predictions.csv
├── src/
│   ├── ML.py
│   └── DL.ipynb
├── .gitignore
├── README.md
└── requirements.txt
```

## Classical ML Pipeline

`src/ML.py` applies lightweight normalization by lowercasing text and collapsing whitespace. It fits a TF-IDF vectorizer on the training set only, using unigram and bigram features with `min_df=2`, then trains balanced logistic regression with `max_iter=1000`.

The selected validation or test split is transformed with the fitted vectorizer. Evaluation data is never used to fit TF-IDF.

## Deep Learning Pipeline

`src/DL.ipynb` uses:

- Vietnamese word segmentation with Underthesea
- 400-dimensional pretrained Vietnamese Word2Vec vectors
- a trainable embedding layer
- a 128-unit LSTM classifier
- class-weighted cross-entropy based only on training labels

The vocabulary and sequence-length statistics are computed from training text only. In the current run, the training vocabulary contains 4,011 tokens, pretrained embedding coverage is 35.2%, and the 95th-percentile sequence length gives `MAX_LEN=27`.

## Class Imbalance

Neutral accounts for only 458 of 11,426 training samples. Both models use training-derived class balancing: logistic regression uses `class_weight="balanced"`, while the LSTM loss uses weights computed from training-label counts. Macro F1 and Neutral-class metrics are reported alongside accuracy so performance on the minority class remains visible.

## Validation Strategy

- **Training set:** used to fit model parameters, TF-IDF, vocabulary, sequence-length settings, and class weights.
- **Validation set:** used for model development, error analysis, and ML-vs-DL comparison.
- **Test set:** reserved for final aggregate evaluation only.

When `SPLIT="valid"`, both implementations save aligned prediction files with `sample_id`, raw text, true label, predicted label, and correctness. When `SPLIT="test"`, they report metrics and a confusion matrix without saving sample-level predictions or displaying individual mistakes.

## Final Test Results

| Model | Representation | Accuracy | Macro F1 | Weighted F1 | Neutral Recall |
|---|---|---:|---:|---:|---:|
| Logistic Regression | TF-IDF (1-2 grams) | 0.8645 | 0.7147 | 0.8705 | **0.4132** |
| LSTM | Underthesea + pretrained Word2Vec | **0.8879** | **0.7406** | **0.8867** | 0.3772 |

Per-class final test performance:

| Model | Class | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| Logistic Regression | Negative | 0.8665 | 0.9120 | 0.8887 | 1,409 |
| Logistic Regression | Neutral | 0.2974 | **0.4132** | 0.3459 | 167 |
| Logistic Regression | Positive | **0.9531** | 0.8698 | 0.9096 | 1,590 |
| LSTM | Negative | **0.9072** | **0.9163** | **0.9117** | 1,409 |
| LSTM | Neutral | **0.4118** | 0.3772 | **0.3937** | 167 |
| LSTM | Positive | 0.9164 | **0.9164** | **0.9164** | 1,590 |

The LSTM pipeline achieved higher test accuracy, macro F1, weighted F1, and per-class F1 for all three classes. Logistic regression retained higher Neutral recall and Positive precision. The models therefore trade off which Neutral samples they recover rather than one model dominating every metric.

## ML vs DL Analysis

The sample-level comparison below uses validation predictions only. On validation, logistic regression reached 0.8806 accuracy and 0.7500 macro F1; the LSTM reached 0.9097 accuracy and 0.7859 macro F1.

| Comparison | Validation samples |
|---|---:|
| Both correct | 1,347 |
| ML wrong, DL correct | 93 |
| ML correct, DL wrong | 47 |
| Both wrong | 96 |

The LSTM made fewer validation errors overall. For Neutral, however, logistic regression made 31 errors and the LSTM made 33. Logistic regression therefore had higher validation Neutral recall (0.5753 vs. 0.5479), while the LSTM had higher Neutral F1 (0.4938 vs. 0.4308). The original expectation that ML missed more Neutral validation samples was not supported by the saved predictions.

Some validation examples illustrate the complementary behavior:

- **ML wrong, DL correct — Neutral:** `hiện nay theo em nhận thấy , trong khi lúc học thì sinh viên chỉ được học về doubledot kỹ năng ghi chép , kỹ tổ chức cuộc họp .` Logistic regression predicted Negative; the LSTM predicted Neutral. This is consistent with a possible benefit from token order and sequential information, although the difference cannot be attributed solely to the LSTM because the DL pipeline also uses Vietnamese segmentation and pretrained Word2Vec.
- **ML wrong, DL correct — Negative:** `không check mail mỗi khi hỏi bài .` Logistic regression predicted Neutral; the LSTM predicted Negative. The DL pipeline may have benefited from the segmented sequence around the negation.
- **ML correct, DL wrong — Negative:** `giảng viên nói tiếng anh không hay .` Logistic regression predicted Negative; the LSTM predicted Positive. This shows that sparse unigram and bigram cues remain useful and that the neural model is not uniformly better.

## Error Analysis

Both models misclassified the validation phrase `sự nhiệt tình .` as Positive even though its label is Neutral. The phrase is short, context-poor, and contains wording commonly associated with positive sentiment, making it difficult to classify from text alone. This observation describes ambiguity in the available text; it does not establish a labeling error.

Detailed validation predictions are available in `Outputs/`. No detailed test predictions or test error samples are stored.

## Installation

Python 3.13 is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Running

Run the classical baseline from the project root:

```powershell
python .\src\ML.py
```

Run the LSTM notebook from `src/` so its relative project path resolves consistently:

```powershell
cd .\src
jupyter notebook DL.ipynb
```

Both files default to `SPLIT="valid"`. Change the configuration line to `SPLIT="test"` only for final evaluation. The test branch intentionally skips detailed prediction export and error analysis.

## Pretrained Word2Vec

Download `wiki.vi.model.bin.gz` from [sonvx/word2vecVN](https://github.com/sonvx/word2vecVN) and place it at:

```text
embeddings/wiki.vi.model.bin.gz
```

The embedding file is an external large artifact and is excluded from Git. The trained checkpoint stores the learned embedding weights required by the current model state.

## Technologies

Python, pandas, NumPy, scikit-learn, matplotlib, PyTorch, Gensim, Underthesea, Jupyter, and pretrained Vietnamese Word2Vec.

## Limitations

- Neutral remains difficult because it is a small class and often contains short or ambiguous feedback.
- The DL pipeline differs from the ML baseline in tokenization, pretrained representations, and sequence modeling, so metric differences cannot be attributed to the LSTM alone.
- Pretrained-vector coverage is 35.2% for the training vocabulary.
- Results come from the official fixed splits rather than repeated cross-validation.
- The notebook currently trains on CPU and does not perform checkpoint selection or early stopping.

## Future Work

- Add an ablation study to separate the effects of Vietnamese segmentation, pretrained embeddings, and the LSTM architecture.
- Evaluate contextual Vietnamese encoders while preserving the same validation/test protocol.
- Investigate class-aware objectives and calibration using validation data only.
- Add a small inference entry point for the committed LSTM checkpoint.
