# Vietnamese Student Feedback Sentiment Classification

Sentiment classification trên UIT-VSFC với TF-IDF baseline và LSTM sử dụng Vietnamese word segmentation cùng pretrained Word2Vec.

## Overview

Project này thực hiện sentiment classification cho phản hồi sinh viên tiếng Việt với ba nhãn: **negative**, **neutral** và **positive**. Hai hướng được so sánh gồm một classical ML baseline gọn nhẹ và một neural sequence model, đồng thời giữ riêng các official split gồm train, validation và test.

Repository tuân theo một workflow có thể tái lập: train data dùng để fit model, validation data phục vụ model development và sample-level analysis, còn test data chỉ được dùng một lần cho final aggregate evaluation.

## Dataset

Project sử dụng ba official split dạng CSV của UIT-VSFC trong `dataset/`:

| Split | Negative | Neutral | Positive | Total |
|---|---:|---:|---:|---:|
| Train | 5,325 | 458 | 5,643 | 11,426 |
| Validation | 705 | 73 | 805 | 1,583 |
| Test | 1,409 | 167 | 1,590 | 3,166 |

Mỗi dòng gồm một phản hồi tiếng Việt và một sentiment label. Neutral là class nhỏ nhất trong cả ba split.

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

`src/ML.py` thực hiện preprocessing nhẹ bằng cách lowercase text và chuẩn hóa whitespace. TF-IDF vectorizer chỉ được fit trên train set, sử dụng unigram và bigram với `min_df=2`, sau đó train balanced Logistic Regression với `max_iter=1000`.

Validation hoặc test split được chọn chỉ dùng vectorizer đã fit để transform. Evaluation data không bao giờ được dùng để fit TF-IDF.

## Deep Learning Pipeline

`src/DL.ipynb` xây dựng pipeline gồm:

- Vietnamese word segmentation với Underthesea
- 400-dimensional pretrained Vietnamese Word2Vec
- trainable embedding layer
- 128-unit LSTM classifier
- class-weighted cross-entropy chỉ dựa trên train labels

Vocabulary và các thống kê về sequence length chỉ được tính từ train text. Trong lần chạy hiện tại, training vocabulary có 4,011 tokens, pretrained embedding coverage đạt 35.2%, và percentile 95 của sequence length cho kết quả `MAX_LEN=27`.

## Class Imbalance

Neutral chỉ chiếm 458 trong tổng số 11,426 training samples. Cả hai model đều xử lý class imbalance dựa trên train data: Logistic Regression dùng `class_weight="balanced"`, còn LSTM loss dùng class weight được tính từ số lượng train labels. Macro F1 và các Neutral-class metrics được báo cáo cùng accuracy để thể hiện rõ hiệu quả trên minority class.

## Validation Strategy

- **Training set:** dùng để fit model parameters, TF-IDF, vocabulary, sequence-length settings và class weights.
- **Validation set:** dùng cho model development, error analysis, ML-vs-DL comparison và sample-level analysis.
- **Test set:** chỉ dành cho final aggregate evaluation.

Khi `SPLIT="valid"`, cả hai implementation đều lưu aligned prediction files gồm `sample_id`, raw text, true label, predicted label và kết quả đúng/sai. Khi `SPLIT="test"`, code chỉ báo cáo metrics và confusion matrix, không lưu detailed predictions và không xem từng error sample.

## Final Test Results

| Model | Representation | Accuracy | Macro F1 | Weighted F1 | Neutral Recall |
|---|---|---:|---:|---:|---:|
| Logistic Regression | TF-IDF (1-2 grams) | 0.8645 | 0.7147 | 0.8705 | **0.4132** |
| LSTM | Underthesea + pretrained Word2Vec | **0.8879** | **0.7406** | **0.8867** | 0.3772 |

Kết quả final test theo từng class:

| Model | Class | Precision | Recall | F1 | Support |
|---|---|---:|---:|---:|---:|
| Logistic Regression | Negative | 0.8665 | 0.9120 | 0.8887 | 1,409 |
| Logistic Regression | Neutral | 0.2974 | **0.4132** | 0.3459 | 167 |
| Logistic Regression | Positive | **0.9531** | 0.8698 | 0.9096 | 1,590 |
| LSTM | Negative | **0.9072** | **0.9163** | **0.9117** | 1,409 |
| LSTM | Neutral | **0.4118** | 0.3772 | **0.3937** | 167 |
| LSTM | Positive | 0.9164 | **0.9164** | **0.9164** | 1,590 |

LSTM pipeline đạt test accuracy, macro F1, weighted F1 và per-class F1 cao hơn ở cả ba class. Logistic Regression vẫn có Neutral Recall và Positive Precision cao hơn. Vì vậy, hai model có sự đánh đổi trong khả năng nhận diện các Neutral samples; không có model nào vượt trội ở mọi metric.

## ML vs DL Analysis

Phần sample-level comparison dưới đây chỉ sử dụng validation predictions. Trên validation set, Logistic Regression đạt accuracy 0.8806 và macro F1 0.7500; LSTM đạt accuracy 0.9097 và macro F1 0.7859.

| Comparison | Validation samples |
|---|---:|
| Both correct | 1,347 |
| ML wrong, DL correct | 93 |
| ML correct, DL wrong | 47 |
| Both wrong | 96 |

LSTM có ít validation errors hơn khi xét tổng thể. Tuy nhiên, với Neutral, Logistic Regression sai 31 samples còn LSTM sai 33 samples. Do đó, Logistic Regression có validation Neutral Recall cao hơn (0.5753 so với 0.5479), trong khi LSTM có Neutral F1 cao hơn (0.4938 so với 0.4308). Các prediction đã lưu không ủng hộ nhận định ban đầu rằng ML bỏ sót nhiều Neutral validation samples hơn.

Một số validation examples cho thấy hai model có những điểm mạnh khác nhau:

- **ML wrong, DL correct — Neutral:** `hiện nay theo em nhận thấy , trong khi lúc học thì sinh viên chỉ được học về doubledot kỹ năng ghi chép , kỹ tổ chức cuộc họp .` Logistic Regression dự đoán Negative; LSTM dự đoán Neutral. Kết quả này phù hợp với khả năng DL pipeline tận dụng token order và sequential information, nhưng không thể quy improvement hoàn toàn cho LSTM vì pipeline còn sử dụng Vietnamese word segmentation và pretrained Word2Vec.
- **ML wrong, DL correct — Negative:** `không check mail mỗi khi hỏi bài .` Logistic Regression dự đoán Neutral; LSTM dự đoán Negative. DL pipeline có thể đã tận dụng tốt hơn segmented sequence xung quanh cấu trúc phủ định.
- **ML correct, DL wrong — Negative:** `giảng viên nói tiếng anh không hay .` Logistic Regression dự đoán Negative; LSTM dự đoán Positive. Trường hợp này cho thấy sparse unigram và bigram cues vẫn hữu ích, đồng thời DL model không phải lúc nào cũng tốt hơn.

## Error Analysis

Cả hai model đều phân loại cụm từ `sự nhiệt tình .` trong validation set thành Positive dù true label là Neutral. Cụm từ này rất ngắn, thiếu ngữ cảnh và chứa lexical cue thường thiên về positive sentiment, nên khó phân loại nếu chỉ dựa vào text. Nhận xét này chỉ phản ánh tính mơ hồ của text hiện có, không khẳng định đây là labeling error.

Detailed validation predictions được lưu trong `Outputs/`. Project không lưu detailed test predictions hoặc test error samples.

## Installation

Khuyến nghị sử dụng Python 3.13.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Running

Chạy classical ML baseline từ project root:

```powershell
python .\src\ML.py
```

Chạy LSTM notebook từ `src/` để relative project path luôn được resolve nhất quán:

```powershell
cd .\src
jupyter notebook DL.ipynb
```

Cả hai file mặc định dùng `SPLIT="valid"`. Chỉ đổi dòng cấu hình thành `SPLIT="test"` khi thực hiện final evaluation. Test branch chủ động bỏ qua detailed prediction export và error analysis.

## Pretrained Word2Vec

Tải `wiki.vi.model.bin.gz` từ [sonvx/word2vecVN](https://github.com/sonvx/word2vecVN) và đặt file tại:

```text
embeddings/wiki.vi.model.bin.gz
```

Pretrained embedding file là một external artifact có kích thước lớn nên không được commit vào Git. Checkpoint đã train lưu learned embedding weights cần thiết cho model state hiện tại.

## Technologies

Python, pandas, NumPy, scikit-learn, matplotlib, PyTorch, Gensim, Underthesea, Jupyter và pretrained Vietnamese Word2Vec.

## Limitations

- Neutral vẫn là class khó vì đây là minority class và thường chứa feedback ngắn hoặc mơ hồ.
- DL pipeline khác ML baseline ở tokenization, pretrained representations và sequence modeling, nên không thể quy toàn bộ khác biệt về metrics cho riêng LSTM.
- Pretrained-vector coverage đạt 35.2% trên training vocabulary.
- Kết quả được lấy từ fixed official splits thay vì repeated cross-validation.
- Notebook hiện train trên CPU và chưa thực hiện checkpoint selection hoặc early stopping.

## Future Work

- Thực hiện ablation study để tách ảnh hưởng của Vietnamese segmentation, pretrained embeddings và LSTM architecture.
- Đánh giá contextual Vietnamese encoders trong khi vẫn giữ nguyên validation/test protocol.
- Nghiên cứu class-aware objectives và calibration chỉ với validation data.
- Bổ sung một inference entry point nhỏ cho LSTM checkpoint đã commit.
