from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .project_paths import MODELS_DIR, OUTPUTS_DIR, PLOTS_DIR, ROOT_DIR, TEST_DATA_PATH


def get_prediction_scores(model, X_test):
    """Extract probability scores for the positive class and predicted labels."""
    y_pred = model.predict(X_test)
    try:
        y_score = model.predict_proba(X_test)[:, 1]
    except (AttributeError, IndexError):
        y_score = model.decision_function(X_test)
    return y_pred, y_score


def evaluate_model(model_name: str, model, X_test: pd.DataFrame, y_test: pd.Series) -> pd.Series:
    """Compute all required evaluation metrics for one trained model."""
    y_pred, y_score = get_prediction_scores(model, X_test)

    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_score),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1': f1_score(y_test, y_pred, zero_division=0),
        'MCC': matthews_corrcoef(y_test, y_pred),
    }
    return pd.Series(metrics)


def save_confusion_matrix(model_name: str, model, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Save a confusion matrix plot for the model."""
    y_pred, _ = get_prediction_scores(model, X_test)
    cm = confusion_matrix(y_test, y_pred, labels=[1, 0])
    labels = ['Yes', 'No']
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(f'{model_name} Confusion Matrix')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('Actual Label')
    fig.tight_layout()

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    slug = model_name.lower().replace(' ', '_').replace('-', '_')
    fig.savefig(PLOTS_DIR / f'{slug}_confusion_matrix.png', dpi=200)
    plt.close(fig)


def save_roc_curve(model_name: str, model, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """Save a single ROC curve for the model."""
    from sklearn.metrics import roc_curve

    y_pred, y_score = get_prediction_scores(model, X_test)
    fpr, tpr, _ = roc_curve(y_test, y_score)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc_score(y_test, y_score):.3f})', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc='lower right')
    fig.tight_layout()

    slug = model_name.lower().replace(' ', '_').replace('-', '_')
    fig.savefig(PLOTS_DIR / f'{slug}_roc_curve.png', dpi=200)
    plt.close(fig)


def save_model_comparison_charts(results_df: pd.DataFrame) -> None:
    """Create metric comparison charts and a combined ROC plot."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    metric_names = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']
    for metric in metric_names:
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.barplot(data=results_df, x='Model', y=metric, hue='Model', dodge=False, palette='viridis', ax=ax)
        if ax.legend_ is not None:
            ax.legend_.remove()
        ax.set_title(f'{metric} Comparison Across Models')
        ax.set_xlabel('Model')
        ax.set_ylabel(metric)
        ax.tick_params(axis='x', rotation=25)
        fig.tight_layout()
        fig.savefig(PLOTS_DIR / f'{metric.lower()}_comparison.png', dpi=200)
        plt.close(fig)

    test_df = pd.read_csv(TEST_DATA_PATH)
    X_eval = test_df.drop(columns=['y'])
    y_true = test_df['y']

    fig, ax = plt.subplots(figsize=(7, 6))
    for _, row in results_df.iterrows():
        model_name = row['Model']
        slug = model_name.lower().replace(' ', '_').replace('-', '_')
        model = joblib.load(MODELS_DIR / f'{slug}.joblib')
        _, y_score = get_prediction_scores(model, X_eval)
        from sklearn.metrics import roc_curve

        fpr, tpr, _ = roc_curve(y_true, y_score)
        ax.plot(fpr, tpr, label=f'{model_name} (AUC={row["AUC"]:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Model ROC Comparison')
    ax.legend(loc='lower right', bbox_to_anchor=(1.0, 0.0))
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / 'roc_comparison.png', dpi=200)
    plt.close(fig)


def evaluate_all_models(pipelines: dict, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """Evaluate each model pipeline and save reports to the outputs directory."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for model_name, model in pipelines.items():
        row = evaluate_model(model_name, model, X_test, y_test)
        rows.append(row)
        save_confusion_matrix(model_name, model, X_test, y_test)
        save_roc_curve(model_name, model, X_test, y_test)

    results_df = pd.DataFrame(rows).sort_values(['AUC', 'F1', 'MCC'], ascending=False, ignore_index=True)
    results_df.to_csv(OUTPUTS_DIR / 'evaluation_results.csv', index=False)
    save_model_comparison_charts(results_df)
    return results_df


def select_best_model(results_df: pd.DataFrame) -> pd.Series:
    """Choose the strongest model using AUC, F1, recall, and MCC in that order of priority."""
    ordered = results_df.sort_values(['AUC', 'F1', 'Recall', 'MCC'], ascending=False)
    return ordered.iloc[0]
