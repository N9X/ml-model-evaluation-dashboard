from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .project_paths import PROCESSED_DIR, RAW_DATA_PATH, TEST_DATA_PATH


def load_bank_data(path: str = str(RAW_DATA_PATH)) -> pd.DataFrame:
    """Load the original UCI bank marketing dataset while preserving raw data."""
    df = pd.read_csv(path, sep=';')
    df.columns = [col.strip() for col in df.columns]
    return df.copy()


def prepare_dataset(path: str = str(RAW_DATA_PATH), save_processed: bool = True) -> pd.DataFrame:
    """Load dataset and normalize categorical unknown values without altering the raw file."""
    df = load_bank_data(path)
    df = df.copy()

    for col in df.columns:
        if col == 'y':
            continue
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].replace('unknown', np.nan)

    df['y'] = df['y'].replace({'yes': 1, 'no': 0}).astype(int)

    if save_processed:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(PROCESSED_DIR / 'bank_marketing_processed.csv', index=False)

    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a reproducible stratified train/test split while excluding the leakage feature duration."""
    feature_columns = [col for col in df.columns if col not in {'duration', 'y'}]
    X = df[feature_columns].copy()
    y = df['y'].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    return X_train, X_test, y_train, y_test


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create a consistent preprocessing pipeline for both numerical and categorical features."""
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]

    numeric_transformer = Pipeline([
        ('scaler', StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
        ],
        remainder='drop',
    )
    return preprocessor


def save_test_dataset(X_test: pd.DataFrame, y_test: pd.Series, path: str = str(TEST_DATA_PATH)) -> None:
    """Save the held-out test dataset with the target column included for app validation."""
    test_df = X_test.copy()
    test_df['y'] = y_test.astype(int).copy()
    test_df.to_csv(path, index=False)
