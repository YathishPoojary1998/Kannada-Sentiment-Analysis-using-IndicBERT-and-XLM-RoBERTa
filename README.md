# Kannada Sentiment Analysis using IndicBERT & XLM-RoBERTa

## 📌 Overview

This project implements a **sentiment analysis system for Kannada language** using **transformer-based transfer learning models** such as XLM-RoBERTa and IndicBERT.

The model classifies text into:

* **Positive**
* **Neutral**
* **Negative**

The work is based on the research paper:

📄 *"A Transfer Learning Approach to the Analysis of Sentiment for the Kannada Language Based on Indic-BERT and XLM-RoBERTa-Base"* 

---

## 🚀 Key Features

* ✅ Kannada sentiment classification (low-resource language)
* ✅ Transfer learning using multilingual transformers
* ✅ Hugging Face + PyTorch implementation
* ✅ Weights & Biases (WandB) integration
* ✅ Supports training + inference pipelines

---

## 🧠 Model Details

* Base Model: `xlm-roberta-base`
* Fine-tuned for Kannada sentiment classification
* Labels:

  * 0 → Negative
  * 1 → Neutral
  * 2 → Positive

---

## 📊 Results

| Metric    | Value     |
| --------- | --------- |
| Accuracy  | **71.9%** |
| Precision | 70%       |
| Recall    | 68%       |
| RMSE      | 1.069     |

As reported in the paper, the model correctly predicted **599 out of 833 test samples** .

---

## 📂 Dataset

* Source: Hindi IIT-Patna Sentiment Dataset (translated to Kannada)
* Split:

  * Train: 6662 samples
  * Validation: 833 samples
  * Test: 833 samples 

---

## ⚙️ Installation

```bash
git clone https://github.com/<your-username>/kannada-sentiment-analysis.git
cd kannada-sentiment-analysis

```

---

## 📦 Requirements

```
transformers
datasets
pandas
numpy
wandb
indic-nlp-library
torch
```

---

## 🏋️ Training

Run training using:

```bash
python src/sentiment_new.py \
    --do_train \
    --train_data data/train_kan.pickle \
    --eval_data data/valid_kan.pickle
```

### Notes:

* Uses Hugging Face Trainer API 
* Supports hyperparameter sweep via WandB

---

## 🔮 Prediction

Run inference:

```bash
python src/predict.py \
    --do_predict \
    --train_data data/train_kan.pickle \
    --eval_data data/test_kan.pickle
```

Output:

```
outputs/predictions.txt
```

Each line:

```
<text>    <predicted_label>
```

Prediction pipeline implemented in .

---

## 🧩 Workflow

1. Data Collection & Translation
2. Label Encoding
3. Tokenization (XLM-R tokenizer)
4. Model Fine-tuning
5. Evaluation
6. Prediction

---

## 🔍 Example

Input:

```
ಉತ್ತಮ ಲ್ಯಾಪ್‌ಟಾಪ್ ಮತ್ತು ಉತ್ತಮ ಕಾರ್ಯಕ್ಷಮತೆ
```

Output:

```
Positive
```

---
