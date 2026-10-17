# 🎬 IMDb Movie Review Sentiment Analysis
> **Big Data Analytics (BDA) & Machine Learning Engineering Mini Project**  
> Scalable Text Ingestion, NLP Preprocessing, Supervised Machine Learning, Explainable AI (XAI), and Interactive Streamlit Dashboard.
<img width="1807" height="936" alt="image" src="https://github.com/user-attachments/assets/355135a7-b2ad-4a09-afb6-05a97f2c7f7c" />


---

## 📌 Executive Summary

This project implements an end-to-end Big Data Analytics text classification system evaluated on the benchmark **Stanford Large Movie Review Dataset (IMDb)** consisting of 50,000 highly polarized movie reviews (25,000 training, 25,000 testing).

The project resolves a critical flaw in traditional lexicon-based approaches:
- **Baseline (TextBlob Lexicon):** Achieved only **68.72% test accuracy** with a severe **optimistic bias (94.33% positive recall vs. 42.32% negative recall)**, misclassifying the majority of critical reviews due to sarcasm and negation blindness.
- **Improved Production Model (TF-IDF + Calibrated Logistic Regression):** Boosted test accuracy to **89.76% (ROC-AUC: 0.9636)** with **balanced class recall (~89.1% negative, ~90.5% positive)** and sub-2ms real-time inference latency.

---

## 🏗️ Project Architecture

```
                                  [IMDb Reviews (50,000 Text Docs)]
                                                 │
                                                 ▼
                             ┌───────────────────────────────────────┐
                             │  PySpark / Pandas Ingestion Pipeline  │
                             └───────────────────────────────────────┘
                                                 │
                                                 ▼
                             ┌───────────────────────────────────────┐
                             │    Modular Preprocessing & Cleaning   │
                             │  (HTML Stripped, Contractions, Regex) │
                             └───────────────────────────────────────┘
                                                 │
                                                 ▼
                             ┌───────────────────────────────────────┐
                             │      Stratified Train/Test Split      │
                             │        (80% Train / 20% Test)         │
                             └───────────────────────────────────────┘
                                                 │
                                                 ▼
                             ┌───────────────────────────────────────┐
                             │     TF-IDF Sublinear N-Gram (1, 2)    │
                             └───────────────────────────────────────┘
                                                 │
                     ┌───────────────────────────┴───────────────────────────┐
                     ▼                                                       ▼
      ┌─────────────────────────────┐                         ┌─────────────────────────────┐
      │  Baseline: TextBlob Lexicon │                         │ Production: Supervised ML   │
      │  • Rule-based Adjectives    │                         │ • L2 Logistic Regression    │
      │  • Accuracy: 68.72%         │                         │ • Accuracy: 89.76%          │
      │  • Negative Recall: 42.32%  │                         │ • Balanced Recall: 89.1%    │
      └─────────────────────────────┘                         └─────────────────────────────┘
                     │                                                       │
                     └───────────────────────────┬───────────────────────────┘
                                                 │
                                                 ▼
                             ┌───────────────────────────────────────┐
                             │    Streamlit Cinema Web Dashboard     │
                             │   (Live Inference, XAI, Batch CSV)    │
                             └───────────────────────────────────────┘
```

---

## 📂 Project Structure

```text
bda miniproject/
│
├── app.py                         # Production Streamlit Web Dashboard (Tabs: Live, Explorer, Benchmark, Batch, Viva)
├── train_model.py                 # Standalone reproducible model training & benchmarking CLI script
│
├── model/                         # Serialized production artifacts
│   ├── sentiment_pipeline.joblib  # Scikit-learn Pipeline (TF-IDF + Logistic Regression, 0.47 MB)
│   ├── metadata.json              # Model configuration, vocab size, and training parameters
│   └── evaluation_results.json    # Verified empirical holdout test metrics for all models
│
├── src/                           # Modular core engine
│   ├── __init__.py
│   ├── preprocessing.py           # Text cleaning, contraction expansion, input validation
│   ├── model.py                   # Pipeline builder, inference engine, XAI feature attribution
│   └── evaluation.py              # Precision, Recall, F1, ROC-AUC, Confusion Matrix calculators
│
├── data/                          # Dataset directory
│   ├── train.xlsx                 # 25,000 raw IMDb reviews
│   ├── sample_reviews.csv         # Curated 200 review dataset for instant demonstration
│   └── aclImdb/                   # Directory structure for backward-compatibility
│
├── tests/                         # Automated unit & integration tests
│   └── test_pipeline.py           # 9 comprehensive pytest test cases
│
├── IMDB_movie_reviews_BDA (3).ipynb # Clean, updated Jupyter notebook with PySpark & MLlib
├── requirements.txt               # Fully validated and pinned dependencies
└── README.md                      # Comprehensive documentation and viva defense guide
```

---

## 📊 Empirical Model Benchmark Results

All metrics below were computed on a strictly leak-free **holdout test set of 4,981 deduplicated reviews**:

| Model Architecture | Test Accuracy | Macro F1 | ROC-AUC | Negative Recall | Positive Recall | Inference Speed |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (TextBlob Lexicon)** | 68.72% | 0.6625 | N/A | **42.32%** ⚠️ | 94.33% | ~12 ms/doc |
| **Multinomial Naive Bayes** | 86.87% | 0.8687 | 0.9436 | 86.52% | 87.21% | < 1 ms/doc |
| **Logistic Regression (C=1.0)** | 89.22% | 0.8922 | 0.9608 | 88.13% | 90.30% | < 2 ms/doc |
| **Logistic Regression (C=2.5) [PROD]** | **89.76%** | **0.8976** | **0.9636** | **89.06%** | **90.46%** | **< 2 ms/doc** |
| **Calibrated LinearSVC (C=0.5)** | 89.82% | 0.8982 | 0.9644 | 89.90% | 89.74% | < 2 ms/doc |

### Key Diagnostic Insight
> **Why did TextBlob fail?**  
> Rule-based lexicons rely on isolated adjective scores. In film reviews, critical language often contains positive words in negative contexts (*"I thought the cinematography had potential, but the acting was a disaster"*), sarcastic praise (*"Brilliant waste of two hours"*), or double negations (*"Not bad at all"*). TextBlob scored these as Positive, generating **57.7% false positives** on actual negative reviews.  
> The **Supervised TF-IDF + Logistic Regression model** learns actual film vocabulary distributions and bigram negation weights, elevating negative recall from **42.32% to 89.06%** (+46.74 percentage points).

---

## 🚀 Quickstart Guide

### 1. Environment Setup & Dependency Installation
Ensure Python 3.10+ is installed:
```powershell
# Clone or navigate to the project directory
cd "C:\Users\manoh\OneDrive\Desktop\bda miniproject"

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify with Automated Test Suite
Run the 9 unit and integration tests:
```powershell
python -m pytest -v
```

### 3. Launch the Streamlit Web Application
Launch the interactive cinema dashboard:
```powershell
streamlit run app.py
```
The application will open automatically in your browser at `http://localhost:8501`.

### 4. (Optional) Retrain the Machine Learning Pipeline
To re-run the benchmark and regenerate model artifacts from scratch:
```powershell
python train_model.py --data-path data/train.xlsx --max-features 25000
```

---

## 🖥️ Streamlit Application Features

1. **✍️ Live Review Analyzer:**
   - Real-time text analysis with character and word counters.
   - Quick example selector (Acclaimed Masterpiece, Critical Disaster, Negation Challenge, Sarcastic).
   - Side-by-side mode comparing Supervised ML vs. TextBlob.
   - **Explainable AI (XAI):** Visual chips highlighting specific n-grams driving Positive or Negative predictions.
   - Text preprocessing inspector (original vs. normalized).
2. **📚 Dataset Explorer:**
   - Class balance donut chart and review character length distributions.
   - Keyword search and length filters with instant preview.
3. **📊 Model Benchmarking:**
   - Empirical test set performance tables and comparative bar charts.
   - Confusion matrices comparing TextBlob vs. Supervised ML.
   - Error breakdown explaining why lexicon matching collapsed on negative reviews.
4. **📁 Batch CSV Inference:**
   - Drag-and-drop CSV upload with auto-detection of text columns.
   - Instant download button for pre-packaged `data/sample_reviews.csv` (200 sample reviews).
   - Sentiment distribution pie/bar charts and export of scored CSV with probabilities.

---

## 🎓 Final-Year Viva Defense Questions & Answers

### Q1: What is the core difference between lexicon-based and supervised machine learning approaches in NLP?
- **Lexicon-Based (TextBlob):** Uses an un-trained dictionary of words with pre-assigned polarity weights. It is zero-shot, fast, but blind to domain-specific context, film jargon, sarcasm, and complex syntax.
- **Supervised ML (TF-IDF + LogReg):** Learns statistical word and phrase associations directly from labeled training data. It captures bigrams (*'not good'*, *'waste of time'*) and weights words based on their empirical correlation with ground truth labels.

### Q2: Why did you use Sublinear TF scaling in your TF-IDF vectorizer?
Sublinear TF scales term frequency logarithmically:  
$$\text{sublinear\_tf} = 1 + \log(\text{tf}) \quad \text{for } \text{tf} > 0$$  
In movie reviews, a user repeating the word *"terrible"* 10 times is not 10 times as negative as someone using it once. Sublinear scaling prevents repetitive words from disproportionately skewing the feature vector.

### Q3: What is the PySpark Python UDF bottleneck and how is it addressed in BDA?
Standard PySpark Python UDFs require data to be serialized from the JVM, transmitted via inter-process communication (IPC) to a Python worker daemon, evaluated, and serialized back to the JVM. This causes high CPU overhead and prevents Spark's Catalyst Optimizer from optimizing execution.  
In production Big Data pipelines, this is resolved by using **Spark MLlib native transformers** or **Vectorized Pandas UDFs (PyArrow)**. For real-time single-sample inference, a lightweight Scikit-Learn pipeline is deployed.

### Q4: How was data leakage prevented during development?
1. Removed 96 duplicate reviews present in the raw data prior to splitting.
2. Executed a **Stratified 80/20 train/test split** with a fixed random seed (`random_state=42`).
3. Fitted the TF-IDF vocabulary **strictly on the 80% training set**; the test set was only transformed, ensuring zero exposure to test vocabulary during training.

### Q5: How is Explainable AI (XAI) implemented in this application?
Because Logistic Regression is a linear classifier, the decision logit is the dot product of feature weights and TF-IDF values:  
$$\text{logit} = \sum (w_i \times \text{tfidf}_i) + b$$  
For any user review, the application calculates the exact contribution of each non-zero n-gram, extracts the highest positive and negative contributors, and displays them as interactive badges.
