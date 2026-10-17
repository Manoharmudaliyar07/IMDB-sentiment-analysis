"""
Text Preprocessing Module for IMDB Sentiment Analysis.
Provides consistent text cleaning, normalization, and validation across
both offline training and live production inference.
"""

import re
from typing import Optional, Tuple

# Common English contraction mappings to preserve critical sentiment and negation cues
CONTRACTION_MAP = {
    r"\bcan\'t\b": "can not",
    r"\bcannot\b": "can not",
    r"\bwon\'t\b": "will not",
    r"\bn\'t\b": " not",
    r"\bi\'m\b": "i am",
    r"\b\'re\b": " are",
    r"\b\'s\b": " is",
    r"\b\'d\b": " would",
    r"\b\'ll\b": " will",
    r"\b\'ve\b": " have",
}


def clean_text(text: Optional[str], expand_contractions: bool = True) -> str:
    """
    Standardize raw review text:
    1. Handle None / non-string types safely.
    2. Lowercase text.
    3. Remove HTML tags (<br />, <p>, etc.).
    4. Expand common contractions (e.g. didn't -> did not) to protect negation cues.
    5. Strip URLs and email patterns.
    6. Remove non-alphabetical characters (punctuation, digits).
    7. Normalize multi-spaces and strip leading/trailing whitespace.
    """
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)

    # 1. Lowercase
    text = text.lower()

    # 2. Strip HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # 3. Strip URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # 4. Expand contractions
    if expand_contractions:
        for pattern, replacement in CONTRACTION_MAP.items():
            text = re.sub(pattern, replacement, text)

    # 5. Remove non-letters (preserve spaces)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # 6. Collapse multiple whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def validate_input_text(text: Optional[str]) -> Tuple[bool, str]:
    """
    Validate review input before inference.
    Returns (is_valid, error_message).
    """
    if text is None:
        return False, "Review text cannot be empty."

    cleaned = clean_text(text)
    if not cleaned:
        return False, "Review contains no alphabetic words or valid text."

    words = cleaned.split()
    if len(words) < 2:
        return False, "Review is too short. Please provide at least 2 meaningful words."

    if len(text) > 20000:
        return False, "Review text is exceptionally long (>20,000 characters). Please provide a concise review."

    return True, ""
