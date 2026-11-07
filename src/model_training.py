from __future__ import annotations

from typing import Dict

import joblib
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .data_preprocessing import build_preprocessor, save_test_dataset, split_data
from .project_paths import MODELS_DIR, ROOT_DIR, TEST_DATA_PATH


def get_model_definitions() -> Dict[str, object]:
    """Return the configured model objects for all five classifiers."""
    return {
        'Logistic Regression': LogisticRegression(
            max_iter=2000,
            class_weight='balanced',
            random_state=42,
            solver='liblinear',
        ),
        'Decision Tree': DecisionTreeClassifier(
            random_state=42,
            max_depth=6,
            min_samples_leaf=20,
            class_weight='balanced',
        ),
        'K-Nearest Neighbors': KNeighborsClassifier(
            n_neighbors=15,
            weights='distance',
            p=2,
        ),
        'Gaussian Naive Bayes': GaussianNB(),
        'Support Vector Machine': SVC(
            kernel='rbf',
            C=2.0,
            gamma='scale',
            class_weight='balanced',
            random_state=42,
        ),
    }


def train_and_save_models(X_train, y_train, X_test, y_test):
    """Train all five pipelines, save the joblib artifacts, and create the reproducible test CSV."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    pipelines = {}

    for model_name, estimator in get_model_definitions().items():
        pipeline = Pipeline([
            ('preprocessor', build_preprocessor(X_train)),
            ('classifier', estimator),
        ])
        pipeline.fit(X_train, y_train)
        slug = model_name.lower().replace(' ', '_').replace('-', '_')
        joblib.dump(pipeline, MODELS_DIR / f'{slug}.joblib')
        pipelines[model_name] = pipeline

    save_test_dataset(X_test, y_test, str(TEST_DATA_PATH))
    return pipelines
