"""
Unit and Integration Tests for IMDB Sentiment Pipeline.
Verifies preprocessing, model loading, probability calibration, explainability,
and edge case handling.
"""

import os
import pytest
import numpy as np

from src.preprocessing import clean_text, validate_input_text
from src.model import (
    load_pipeline_artifacts,
    predict_sentiment_ml,
    predict_textblob_baseline,
    explain_prediction,
)
from src.evaluation import compute_metrics


def test_clean_text_basic():
    raw = "<br />This movie was CANNOT be missed! Visit http://imdb.com for details."
    cleaned = clean_text(raw)
    assert "<br />" not in cleaned
    assert "http" not in cleaned
    assert "cannot" not in cleaned or "can not" in cleaned
    assert cleaned.islower()


def test_clean_text_edge_cases():
    assert clean_text("") == ""
    assert clean_text(None) == ""
    assert clean_text("12345 67890 !@#$%") == ""


def test_validate_input_text():
    # Valid review
    valid, msg = validate_input_text("Excellent film with stunning cinematography.")
    assert valid is True
    assert msg == ""

    # Empty
    valid, msg = validate_input_text("")
    assert valid is False

    # None
    valid, msg = validate_input_text(None)
    assert valid is False

    # Single character / single word
    valid, msg = validate_input_text("a")
    assert valid is False


def test_model_loading():
    pipeline, metadata = load_pipeline_artifacts("model/sentiment_pipeline.joblib", "model/metadata.json")
    assert pipeline is not None, "Production pipeline artifact not found."
    assert metadata is not None, "Metadata artifact not found."
    assert "classes" in metadata
    assert metadata["vocabulary_size"] > 5000


def test_supervised_ml_positive_prediction():
    pipeline, _ = load_pipeline_artifacts("model/sentiment_pipeline.joblib")
    assert pipeline is not None
    review = "An extraordinary masterpiece! Brilliant acting, captivating script, and breathtaking direction."
    result = predict_sentiment_ml(pipeline, review)

    assert result["success"] is True
    assert result["label"] == "Positive"
    assert result["prob_positive"] > 0.60
    assert result["confidence"] > 0.60
    assert len(result["pos_tokens"]) > 0


def test_supervised_ml_negative_prediction():
    pipeline, _ = load_pipeline_artifacts("model/sentiment_pipeline.joblib")
    assert pipeline is not None
    review = "Absolute garbage. Awful acting, terrible plot, and a complete waste of two hours."
    result = predict_sentiment_ml(pipeline, review)

    assert result["success"] is True
    assert result["label"] == "Negative"
    assert result["prob_negative"] > 0.60
    assert len(result["neg_tokens"]) > 0


def test_explain_prediction_feature_attribution():
    pipeline, _ = load_pipeline_artifacts("model/sentiment_pipeline.joblib")
    assert pipeline is not None
    cleaned = "superb acting but awful direction and boring plot"
    pos_tokens, neg_tokens = explain_prediction(pipeline, cleaned)

    assert isinstance(pos_tokens, list)
    assert isinstance(neg_tokens, list)
    pos_words = [t[0] for t in pos_tokens]
    neg_words = [t[0] for t in neg_tokens]

    assert "superb" in pos_words
    assert any(w in neg_words for w in ["awful", "boring", "direction"])


def test_textblob_baseline():
    pos_rev = "I loved this wonderful movie so much!"
    res_pos = predict_textblob_baseline(pos_rev)
    assert res_pos["success"] is True
    assert res_pos["label"] == "Positive"
    assert res_pos["polarity"] > 0

    neg_rev = "This is the worst and most boring film ever made."
    res_neg = predict_textblob_baseline(neg_rev)
    assert res_neg["success"] is True
    assert res_neg["label"] == "Negative"
    assert res_neg["polarity"] < 0


def test_compute_metrics():
    y_true = np.array([1, 1, 0, 0, 1])
    y_pred = np.array([1, 0, 0, 0, 1])
    y_proba = np.array([0.9, 0.4, 0.1, 0.2, 0.8])

    metrics = compute_metrics(y_true, y_pred, y_proba)
    assert metrics["accuracy"] == 0.8
    assert metrics["confusion_matrix"] == [[2, 0], [1, 2]]
    assert metrics["roc_auc"] > 0.8
