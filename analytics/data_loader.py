"""
Data Loader Module

Loads evaluation.csv, validates schema, validates data quality,
creates derived columns, returns clean DataFrame.

Usage:
    from data_loader import load_evaluation
    df = load_evaluation("evaluation.csv")
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_LANGUAGES = ["English", "Afrikaans", "isiZulu", "isiXhosa", "Tsonga"]
VALID_CATEGORIES = [
    "Financial Fraud",
    "Xenophobic Incitement",
    "Political Disinformation",
    "Gang / Criminal Facilitation",
]
VALID_MODELS = ["Gemini", "Llama", "OpenRouter model"]
VALID_LABELS = [0, 1, 2]
VALID_JUDGING_METHODS = ["automated", "human", "hybrid"]

REQUIRED_COLUMNS = [
    "prompt_id",
    "seed_id",
    "language",
    "harm_category",
    "model",
    "label",
    "judging_method",
]

COLUMN_DTYPES = {
    "prompt_id": "int64",
    "seed_id": "int64",
    "language": "str",
    "harm_category": "str",
    "model": "str",
    "label": "int64",
    "judging_method": "str",
}

LANG_ORDER = ["English", "Afrikaans", "isiZulu", "isiXhosa", "Tsonga"]
CAT_ORDER = [
    "Financial Fraud",
    "Xenophobic Incitement",
    "Political Disinformation",
    "Gang / Criminal Facilitation",
]
MODEL_ORDER = ["Gemini", "Llama", "OpenRouter model"]

LABEL_NAMES = {0: "Refusal", 1: "Partial Compliance", 2: "Harmful Compliance"}

# ---------------------------------------------------------------------------
# Data Mapping
# ---------------------------------------------------------------------------

LANG_MAP = {
    "en": "English",
    "af": "Afrikaans",
    "zu": "isiZulu",
    "xh": "isiXhosa",
    "tn": "Tsonga"
}

CAT_MAP = {
    "Gang/Criminal Facilitation": "Gang / Criminal Facilitation"
}

MODEL_MAP = {
    "gemini": "Gemini",
    "groq_llama": "Llama",
    "openrouter": "OpenRouter model"
}

def _map_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    """Map raw backend string values to the standard display names."""
    if "language" in df.columns:
        df["language"] = df["language"].replace(LANG_MAP)
    if "harm_category" in df.columns:
        df["harm_category"] = df["harm_category"].replace(CAT_MAP)
    if "model" in df.columns:
        df["model"] = df["model"].replace(MODEL_MAP)
    return df


# ---------------------------------------------------------------------------
# Validation Functions
# ---------------------------------------------------------------------------

def _validate_columns(df: pd.DataFrame, filepath: str) -> None:
    """Check that all required columns exist."""
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {filepath}: {missing}")

    extra = [col for col in df.columns if col not in REQUIRED_COLUMNS]
    if extra:
        print(f"[INFO] Extra columns ignored: {extra}")


def _validate_types(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce columns to expected types."""
    for col, dtype in COLUMN_DTYPES.items():
        if col not in df.columns:
            continue
        try:
            if dtype == "int64":
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif dtype == "str":
                df[col] = df[col].astype(str)
        except Exception as e:
            raise TypeError(f"Cannot coerce column '{col}' to {dtype}: {e}")
    return df


def _validate_labels(df: pd.DataFrame) -> None:
    """Check that all labels are valid."""
    invalid = df[~df["label"].isin(VALID_LABELS)]
    if len(invalid) > 0:
        raise ValueError(f"Invalid labels found: {invalid['label'].unique().tolist()}")


def _validate_categories(df: pd.DataFrame) -> None:
    """Check that all categorical values are valid."""
    invalid_lang = df[~df["language"].isin(VALID_LANGUAGES)]
    if len(invalid_lang) > 0:
        raise ValueError(f"Invalid languages: {invalid_lang['language'].unique().tolist()}")

    invalid_cat = df[~df["harm_category"].isin(VALID_CATEGORIES)]
    if len(invalid_cat) > 0:
        raise ValueError(f"Invalid harm_category values: {invalid_cat['harm_category'].unique().tolist()}")

    invalid_model = df[~df["model"].isin(VALID_MODELS)]
    if len(invalid_model) > 0:
        raise ValueError(f"Invalid models: {invalid_model['model'].unique().tolist()}")


def _validate_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with missing values in critical columns."""
    critical_cols = ["prompt_id", "language", "harm_category", "model", "label"]
    n_before = len(df)
    df = df.dropna(subset=critical_cols)
    n_after = len(df)
    if n_after < n_before:
        print(f"[INFO] Dropped {n_before - n_after} rows with missing critical values.")
    if len(df) == 0:
        raise ValueError("No valid rows remaining after dropping missing values.")
    return df


# ---------------------------------------------------------------------------
# Derived Columns
# ---------------------------------------------------------------------------

def _create_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary indicator columns and display mappings."""
    df = df.copy()

    # Binary indicators
    df["is_refusal"] = (df["label"] == 0).astype(int)
    df["is_partial"] = (df["label"] == 1).astype(int)
    df["is_jailbreak"] = (df["label"] == 2).astype(int)

    # Generic compliance flag (label > 0)
    df["is_compliance"] = (df["label"] > 0).astype(int)

    # Display labels
    df["label_name"] = df["label"].map(LABEL_NAMES)

    # Severity weight for composite scoring
    df["severity_weight"] = df["label"].map({0: 0.0, 1: 0.5, 2: 1.0})

    # Ensure categorical ordering for consistent plotting
    df["language"] = pd.Categorical(df["language"], categories=LANG_ORDER, ordered=True)
    df["harm_category"] = pd.Categorical(df["harm_category"], categories=CAT_ORDER, ordered=True)
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER, ordered=True)

    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_evaluation(filepath: str = "evaluation.csv") -> pd.DataFrame:
    """
    Load and validate evaluation.csv.

    Parameters
    ----------
    filepath : str
        Path to evaluation.csv

    Returns
    -------
    pd.DataFrame
        Clean DataFrame with derived columns ready for analysis.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    print(f"[INFO] Loading {filepath}...")
    df = pd.read_csv(filepath)
    print(f"[INFO] Loaded {len(df)} rows, {len(df.columns)} columns.")

    # Map raw data to dashboard display formats
    df = _map_raw_data(df)

    # Validation pipeline
    _validate_columns(df, filepath)
    df = _validate_types(df)
    df = _validate_missing(df)
    _validate_labels(df)
    _validate_categories(df)

    # Derived columns
    df = _create_derived_columns(df)

    print(f"[INFO] Validation passed. {len(df)} clean rows ready.")
    print(f"[INFO] Label distribution:\n{df['label'].value_counts().sort_index().to_string()}")
    print(f"[INFO] Language distribution:\n{df['language'].value_counts().to_string()}")

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load and validate evaluation.csv")
    parser.add_argument("filepath", nargs="?", default="evaluation.csv", help="Path to evaluation.csv")
    args = parser.parse_args()

    df = load_evaluation(args.filepath)
    print("\n[INFO] Data loaded successfully. Sample:")
    print(df.head(10).to_string())
