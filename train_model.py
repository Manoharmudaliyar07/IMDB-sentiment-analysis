"""
Offline Model Training and Evaluation Script.
Trains leak-free supervised models on the IMDB sentiment dataset,
evaluates against the TextBlob rule-based baseline, and serializes production artifacts.

Usage:
    python train_model.py
    python train_model.py --data-path data/train.xlsx --max-features 25000
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

# Ensure utf-8 stdout encoding on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from textblob import TextBlob

from src.preprocessing import clean_text
from src.evaluation import compute_metrics, format_metrics_table
from src.model import build_pipeline, save_pipeline_artifacts


def load_raw_dataset(data_path: str) -> pd.DataFrame:
    """Load dataset from Excel, CSV, or raw folder structure."""
    if os.path.isfile(data_path):
        if data_path.endswith(".xlsx"):
            df = pd.read_excel(data_path)
        else:
            df = pd.read_csv(data_path)

        # Standardize column names
        text_col = next((c for c in df.columns if c.lower() in ["reviews", "review", "text"]), df.columns[0])
        label_col = next((c for c in df.columns if c.lower() in ["sentiment", "label", "target"]), df.columns[1])
        df = df[[text_col, label_col]].rename(columns={text_col: "review", label_col: "sentiment"})
    elif os.path.isdir(data_path):
        rows = []
        for lab in ["pos", "neg"]:
            folder = os.path.join(data_path, lab)
            if os.path.isdir(folder):
                for fn in os.listdir(folder):
                    fp = os.path.join(folder, fn)
                    if os.path.isfile(fp):
                        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                            rows.append((f.read(), "pos" if lab == "pos" else "neg"))
        df = pd.DataFrame(rows, columns=["review", "sentiment"])
    else:
        raise FileNotFoundError(f"Cannot find dataset at '{data_path}'.")

    # Normalize labels to binary 0/1
    df["target"] = df["sentiment"].astype(str).str.lower().map(
        lambda s: 1 if "pos" in s or s == "1" else 0
    )
    return df


def main():
    parser = argparse.ArgumentParser(description="Train and benchmark IMDB sentiment models.")
    parser.add_argument("--data-path", type=str, default="data/train.xlsx", help="Path to dataset file or dir.")
    parser.add_argument("--test-size", type=float, default=0.20, help="Holdout test proportion.")
    parser.add_argument("--random-state", type=int, default=42, help="Reproducibility random seed.")
    parser.add_argument("--max-features", type=int, default=25000, help="Max vocabulary size for TF-IDF.")
    parser.add_argument("--output-dir", type=str, default="model", help="Directory for output artifacts.")
    args = parser.parse_args()

    print("=" * 70)
    print("[INFO] IMDB Sentiment Analysis Pipeline Training")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # 1. Load data
    print(f"\n[1/6] Loading dataset from: {args.data_path}")
    df = load_raw_dataset(args.data_path)
    initial_count = len(df)
    print(f"  Loaded {initial_count:,} records.")

    # 2. Data Cleaning & Deduplication
    print("\n[2/6] Cleaning text and eliminating duplicate records...")
    df = df.drop_duplicates(subset=["review"]).reset_index(drop=True)
    dedup_count = len(df)
    if initial_count != dedup_count:
        print(f"  Removed {initial_count - dedup_count} duplicate samples. Unique remaining: {dedup_count:,}")

    df["clean_review"] = df["review"].map(clean_text)
    # Remove records that became empty after cleaning
    df = df[df["clean_review"].str.len() > 5].reset_index(drop=True)
    print(f"  Cleaned dataset ready: {len(df):,} reviews (50/50 balance checked).")

    # 3. Stratified Train / Test Split
    print(f"\n[3/6] Splitting data ({int((1-args.test_size)*100)}% Train / {int(args.test_size*100)}% Test) with stratify...")
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_review"],
        df["target"].values,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df["target"].values,
    )
    print(f"  Train samples: {len(X_train):,} | Test samples: {len(X_test):,}")

    # 4. Evaluate TextBlob Baseline on Test Set
    print("\n[4/6] Evaluating TextBlob Rule-Based Lexicon Baseline on Test Set...")
    tb_preds = []
    # Sample up to 2500 test records for speedy baseline evaluation
    tb_eval_size = min(len(X_test), 2500)
    tb_sample_X = X_test.iloc[:tb_eval_size]
    tb_sample_y = y_test[:tb_eval_size]
    for text in tb_sample_X:
        pol = TextBlob(text).sentiment.polarity
        tb_preds.append(1 if pol > 0 else 0)

    tb_metrics = compute_metrics(tb_sample_y, np.array(tb_preds), None)

    # 5. Feature Engineering & Supervised Model Comparison
    print(f"\n[5/6] Fitting TF-IDF Vectorizer (max_features={args.max_features}, ngram=(1,2))...")
    tfidf = TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=3,
        strip_accents="unicode",
    )
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec = tfidf.transform(X_test)

    candidate_models = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.5),
        "Logistic Regression (C=1.0)": LogisticRegression(C=1.0, max_iter=1000, random_state=args.random_state),
        "Logistic Regression (C=2.5)": LogisticRegression(C=2.5, max_iter=1000, random_state=args.random_state),
        "Calibrated LinearSVC (C=0.5)": CalibratedClassifierCV(LinearSVC(C=0.5, random_state=args.random_state)),
    }

    all_metrics = {"Baseline (TextBlob Lexicon)": tb_metrics}

    print("\nBenchmarking Candidate Algorithms:")
    for name, clf in candidate_models.items():
        t0 = time.time()
        clf.fit(X_train_vec, y_train)
        fit_duration = time.time() - t0

        y_pred = clf.predict(X_test_vec)
        y_proba = clf.predict_proba(X_test_vec)[:, 1] if hasattr(clf, "predict_proba") else None

        metrics = compute_metrics(y_test, y_pred, y_proba)
        all_metrics[name] = metrics
        print(f"  [OK] {name:<30} | Test Acc: {metrics['accuracy']:.2%} | F1: {metrics['f1_macro']:.4f} | ROC-AUC: {metrics['roc_auc']:.4f} (Fit: {fit_duration:.2f}s)")

    # Print Comparison Table
    print("\n" + "=" * 70)
    print("[TABLE] EMPIRICAL MODEL COMPARISON TABLE (TEST SET)")
    print("=" * 70)
    summary_df = format_metrics_table(all_metrics)
    print(summary_df.to_string(index=False))

    # 6. Build and Save Production Pipeline
    print(f"\n[6/6] Assembling and Serializing Production Model Pipeline...")
    production_pipeline = build_pipeline(
        max_features=args.max_features,
        ngram_range=(1, 2),
        C=2.5,
        random_state=args.random_state,
    )
    print("  Training full end-to-end pipeline on training set...")
    production_pipeline.fit(X_train, y_train)

    metadata = {
        "model_name": "TF-IDF + Calibrated Logistic Regression",
        "algorithm": "LogisticRegression (L2 regularization, C=2.5)",
        "vectorizer": "TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True)",
        "vocabulary_size": len(production_pipeline.named_steps["tfidf"].vocabulary_),
        "classes": ["Negative", "Positive"],
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "test_accuracy": float(all_metrics["Logistic Regression (C=2.5)"]["accuracy"]),
        "test_macro_f1": float(all_metrics["Logistic Regression (C=2.5)"]["f1_macro"]),
        "test_roc_auc": float(all_metrics["Logistic Regression (C=2.5)"]["roc_auc"]),
        "training_date": datetime.now().isoformat(),
        "random_state": args.random_state,
    }

    model_file = os.path.join(args.output_dir, "sentiment_pipeline.joblib")
    meta_file = os.path.join(args.output_dir, "metadata.json")
    eval_file = os.path.join(args.output_dir, "evaluation_results.json")

    save_pipeline_artifacts(production_pipeline, metadata, model_file, meta_file)

    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"  [OK] Production pipeline saved to: {model_file} ({os.path.getsize(model_file) / (1024*1024):.2f} MB)")
    print(f"  [OK] Metadata saved to: {meta_file}")
    print(f"  [OK] Evaluation benchmark results saved to: {eval_file}")
    print("\n[SUCCESS] Training and artifact serialization complete successfully!")


if __name__ == "__main__":
    main()
