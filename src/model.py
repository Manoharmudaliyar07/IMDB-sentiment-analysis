"""
Model Architecture, Pipeline, Inference, and Explainability Module.
Integrates scikit-learn TF-IDF + Calibrated Classifier with TextBlob baseline comparison
and token-level feature attribution.
"""

import os
import json
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from textblob import TextBlob

from src.preprocessing import clean_text, validate_input_text

DEFAULT_MODEL_PATH = os.path.join("model", "sentiment_pipeline.joblib")
DEFAULT_METADATA_PATH = os.path.join("model", "metadata.json")


def build_pipeline(
    max_features: int = 25000,
    ngram_range: Tuple[int, int] = (1, 2),
    C: float = 2.5,
    random_state: int = 42,
) -> Pipeline:
    """
    Construct the production machine learning pipeline:
    TF-IDF Vectorizer with sublinear term-frequency scaling & unigram/bigram features
    coupled with L2-regularized Logistic Regression.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        min_df=3,
        strip_accents="unicode",
    )
    classifier = LogisticRegression(
        C=C,
        max_iter=1000,
        random_state=random_state,
        class_weight="balanced",
        solver="lbfgs",
    )
    return Pipeline([("tfidf", vectorizer), ("clf", classifier)])


def save_pipeline_artifacts(
    pipeline: Pipeline,
    metadata: Dict[str, Any],
    model_path: str = DEFAULT_MODEL_PATH,
    metadata_path: str = DEFAULT_METADATA_PATH,
) -> None:
    """Save trained pipeline and accompanying metadata to disk."""
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path, compress=3)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def load_pipeline_artifacts(
    model_path: str = DEFAULT_MODEL_PATH,
    metadata_path: str = DEFAULT_METADATA_PATH,
) -> Tuple[Optional[Pipeline], Optional[Dict[str, Any]]]:
    """Safely load pipeline and metadata; return (None, None) if not found."""
    if not os.path.isfile(model_path):
        return None, None

    try:
        pipeline = joblib.load(model_path)
        metadata = None
        if os.path.isfile(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        return pipeline, metadata
    except Exception as e:
        print(f"Error loading model artifacts: {e}")
        return None, None


def explain_prediction(
    pipeline: Pipeline, cleaned_text: str, top_k: int = 8
) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    """
    Compute linear model feature attribution for the input review.
    Maps active n-grams to their classifier coefficient weights (w_i * tfidf_i).
    Returns (top_positive_tokens, top_negative_tokens).
    """
    try:
        vectorizer: TfidfVectorizer = pipeline.named_steps["tfidf"]
        classifier: LogisticRegression = pipeline.named_steps["clf"]

        # Transform single input
        vec = vectorizer.transform([cleaned_text])
        feature_names = vectorizer.get_feature_names_out()
        coefs = classifier.coef_[0]

        # Active features (non-zero TF-IDF values in this sample)
        non_zero_indices = vec.nonzero()[1]
        token_scores = []
        for idx in non_zero_indices:
            token = feature_names[idx]
            weight = float(coefs[idx])
            tfidf_val = float(vec[0, idx])
            # Total logit contribution = weight * tfidf
            contribution = weight * tfidf_val
            token_scores.append((token, contribution, weight))

        # Positive tokens drive class 1 (Positive), negative tokens drive class 0 (Negative)
        pos_tokens = sorted(
            [t for t in token_scores if t[1] > 0], key=lambda x: -x[1]
        )[:top_k]
        neg_tokens = sorted(
            [t for t in token_scores if t[1] < 0], key=lambda x: x[1]
        )[:top_k]

        return [(t[0], t[1]) for t in pos_tokens], [(t[0], t[1]) for t in neg_tokens]
    except Exception as e:
        print(f"Feature explanation error: {e}")
        return [], []


def predict_sentiment_ml(
    pipeline: Pipeline, raw_text: str, decision_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Run full inference using the trained Supervised ML pipeline.
    Includes input validation, cleaning, probability estimation, and explainability.
    """
    is_valid, err_msg = validate_input_text(raw_text)
    if not is_valid:
        return {"success": False, "error": err_msg}

    cleaned = clean_text(raw_text)
    if not cleaned:
        return {"success": False, "error": "Cleaned review has no valid tokens."}

    # Predict probability [P(Negative), P(Positive)]
    probabilities = pipeline.predict_proba([cleaned])[0]
    p_neg = float(probabilities[0])
    p_pos = float(probabilities[1])

    label = "Positive" if p_pos >= decision_threshold else "Negative"
    confidence = p_pos if label == "Positive" else p_neg

    pos_tokens, neg_tokens = explain_prediction(pipeline, cleaned)

    return {
        "success": True,
        "label": label,
        "confidence": confidence,
        "prob_positive": p_pos,
        "prob_negative": p_neg,
        "cleaned_text": cleaned,
        "pos_tokens": pos_tokens,
        "neg_tokens": neg_tokens,
        "model_type": "Supervised ML (TF-IDF + Logistic Regression)",
    }


def predict_textblob_baseline(
    raw_text: str, neutral_band: float = 0.0
) -> Dict[str, Any]:
    """
    Run baseline rule-based sentiment inference using TextBlob lexicon.
    Preserved for direct viva comparison against the supervised model.
    """
    is_valid, err_msg = validate_input_text(raw_text)
    if not is_valid:
        return {"success": False, "error": err_msg}

    cleaned = clean_text(raw_text, expand_contractions=False)
    blob = TextBlob(cleaned)
    polarity = float(blob.sentiment.polarity)
    subjectivity = float(blob.sentiment.subjectivity)

    if polarity > neutral_band:
        label = "Positive"
    elif polarity < -neutral_band:
        label = "Negative"
    else:
        label = "Neutral"

    return {
        "success": True,
        "label": label,
        "polarity": polarity,
        "subjectivity": subjectivity,
        "cleaned_text": cleaned,
        "model_type": "Rule-Based Baseline (TextBlob Lexicon)",
    }
