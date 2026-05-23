# Improving Robustness of Sarcasm for Sentiment Analysis on Twitter

Author: Thu Pham

This is the repository for Master Thesis project on improving Twitter sentiment analysis robustness on sarcastic tweets using Multi-Task Learning (MTL) and Rationale Supervision (RS).

The thesis compares several RoBERTa-based architectures trained on TweetEval and evaluated both on the standard test set and a sarcasm-heavy out-of-distribution (OOD) golden set.

---

## Models

| Model | Description |
|---|---|
| Baseline | Standard single-task sentiment classifier |
| MTL | Joint sentiment + sarcasm detection |
| RS | Uses LLM-generated rationales as auxiliary supervision |
| Combined | Combines sarcasm detection and rationale supervision together |

All models use:
- `cardiffnlp/twitter-roberta-base-sentiment-latest`
- Mean pooling
- Layer-wise learning rate decay (LLRD)
- Multi-seed evaluation (3 seeds)

---

## Project Structure

```bash
Thesis-Twitter-Sentiment-Analysis/
├─ data/
│  ├─ raw/                # Original TweetEval and iSarcasm datasets
│  ├─ processed/          # Processed and rationale-augmented datasets
│  └─ batch_api_calls/    # JSONL files for LLM rationale generation
│
├─ models/                # Saved model checkpoints (.pt)
├─ notebooks/             # EDA, error analysis, and pilot experiments
├─ outputs/               # Predictions, figures, logs, and evaluation results
│
├─ src/
│  ├─ baseline_model.py   # Standard RoBERTa (1 head: sentiment)
│  ├─ mtl_model.py        # Multi-task RoBERTa (2 heads: sentiment + sarcasm)
│  ├─ rs_model.py         # Rationale-supervised RoBERTa (2 heads: sentiment + rationale)
│  ├─ combined_model.py   # Combined model (3 heads: sentiment + sarcasm + rationale)
│  │
│  ├─ config.py           # Hyperparameters, loss weights, paths, and API keys
│  ├─ preprocessing.py    # Data cleaning and preprocessing pipeline
│  ├─ data_loader.py      # Loads tweets, labels, and rationale embeddings
│  ├─ evaluation.py       # Evaluation metrics and reporting functions
│  └─ utils.py            # Helper functions and masked loss implementation
│
├─ run_baseline.py        # Train and evaluate Baseline model
├─ run_mtl.py             # Train and evaluate MTL model
├─ run_rs.py              # Train and evaluate RS model
├─ run_combined.py        # Train and evaluate Combined model
│
├─ run_mtl_tuning.py      # Hyperparameter tuning for MTL
├─ run_rs_tuning.py       # Hyperparameter tuning for RS
├─ run_combined_tuning.py # Hyperparameter tuning for Combined
│
├─ generate_rationales.py # Generate input files for Batch OpenAI API
│
├─ README.md
└─ requirements.txt
```

---
## How to run
### Setup

Clone the repository and install dependencies:

```bash
git clone https://github.com/catherine0911/Thesis-Twitter-Sentiment-Analysis.git
cd Thesis
pip install -r requirements.txt
```

### Train Models

Run each model separately:

```bash
python run_baseline.py
python run_mtl.py
python run_rs.py
python run_combined_model.py
```

### Hyperparameter Tuning

Optional Optuna tuning scripts:

```bash
python run_mtl_tuning.py
python run_rs_tuning.py
python run_combined_tuning.py
```
---

## Main Results

| Model | Test F1 | Gold F1 (OOD) |
|---|---|---|
| Baseline | 0.7192 ± 0.0031 | 0.5848 ± 0.0304 |
| MTL | 0.7158 ± 0.0025 | 0.6429 ± 0.0174 |
| RS | 0.7186 ± 0.0057 | 0.7010 ± 0.0075 |
| Combined | 0.7165 ± 0.0031 | **0.7244 ± 0.0350** |

The results show that rationale supervision improves robustness on sarcastic tweets substantially, while the Combined model achieves the best overall OOD performance.

---

## Acknowledgements

### Backbone Model
- `cardiffnlp/twitter-roberta-base-sentiment-latest` Link: https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest

### Sentence Embedding Model
- `sentence-transformers/all-MiniLM-L6-v2` Link: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

### Datasets
- TweetEval (Barbieri et al., 2020) Link: https://github.com/cardiffnlp/tweeteval
- iSarcasmEval (Abu Farha et al., 2022) Link: https://github.com/dmbavkar/iSarcasm
- SemEval-2014 Task 9 sarcasm dataset (Rosenthal et al., 2014) Link: https://alt.qcri.org/semeval2014/task9/index.php?id=data-and-tools