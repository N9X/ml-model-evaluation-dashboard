from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.project_paths import MODELS_DIR, ROOT_DIR, TEST_DATA_PATH

MODEL_OPTIONS = [
    'Logistic Regression',
    'Decision Tree',
    'K-Nearest Neighbors',
    'Gaussian Naive Bayes',
    'Support Vector Machine',
]


def slugify_model_name(name: str) -> str:
    return name.lower().replace(' ', '_').replace('-', '_')


def get_model_path(name: str) -> Path:
    return MODELS_DIR / f'{slugify_model_name(name)}.joblib'


def read_uploaded_csv(uploaded_file) -> pd.DataFrame:
    """Read comma- or semicolon-delimited CSV files and normalize column names."""
    df = pd.read_csv(uploaded_file, sep=None, engine='python')
    df.columns = [str(column).strip().lstrip('\ufeff') for column in df.columns]
    return df


def load_all_models() -> dict[str, object]:
    models = {}
    for name in MODEL_OPTIONS:
        path = get_model_path(name)
        if not path.exists():
            raise FileNotFoundError(f'Model file not found: {path}')
        models[name] = joblib.load(path)
    return models


def validate_uploaded_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if df.empty:
        raise ValueError('The uploaded CSV is empty.')

    expected_features = pd.read_csv(TEST_DATA_PATH).columns.drop('y').tolist()
    missing = [col for col in expected_features if col not in df.columns]
    if missing:
        raise ValueError(f'Missing required columns: {missing}')

    if 'y' not in df.columns:
        raise ValueError('The uploaded dataset must include the target column "y".')

    # duration is present in the original UCI CSV but is intentionally not a model feature.
    if 'duration' in df.columns:
        df = df.drop(columns=['duration'])

    raw_target = df['y'].astype('string').str.strip().str.lower()
    target_mapping = {'yes': 1, 'no': 0, '1': 1, '0': 0}
    target = raw_target.map(target_mapping)
    if target.isna().any():
        invalid_values = raw_target[target.isna()].drop_duplicates().head(5).tolist()
        raise ValueError(
            'The target column "y" must contain only yes/no or 1/0 values. '
            f'Invalid value(s): {invalid_values}'
        )

    feature_frame = df[expected_features].copy()
    for column in feature_frame.select_dtypes(include=['object', 'string']).columns:
        feature_frame[column] = feature_frame[column].replace('unknown', pd.NA)
    return feature_frame, target.astype(int)


def compute_metrics_for_model(model, X: pd.DataFrame, y_true: pd.Series) -> dict:
    y_pred = model.predict(X)
    try:
        y_score = model.predict_proba(X)[:, 1]
    except (AttributeError, IndexError):
        y_score = model.decision_function(X)

    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'AUC': roc_auc_score(y_true, y_score),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'MCC': matthews_corrcoef(y_true, y_pred),
    }


def render_metric_cards(metrics: dict) -> None:
    cols = st.columns(6)
    metric_items = [
        ('Accuracy', metrics['Accuracy']),
        ('AUC', metrics['AUC']),
        ('Precision', metrics['Precision']),
        ('Recall', metrics['Recall']),
        ('F1', metrics['F1']),
        ('MCC', metrics['MCC']),
    ]
    for col, (label, value) in zip(cols, metric_items):
        col.metric(label, f'{value:.4f}')


def render_confusion_matrix(y_true: pd.Series, y_pred: pd.Series) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    cm_df = pd.DataFrame(cm, index=['Actual Yes', 'Actual No'], columns=['Pred Yes', 'Pred No'])
    fig = px.imshow(cm_df, text_auto=True, color_continuous_scale='Blues', aspect='auto')
    fig.update_layout(title='Confusion Matrix', xaxis_title='Predicted Label', yaxis_title='Actual Label')
    st.plotly_chart(fig, use_container_width=True)

    tp = cm[0, 0]
    tn = cm[1, 1]
    fp = cm[1, 0]
    fn = cm[0, 1]
    st.caption(f'True Positive: {tp} | True Negative: {tn} | False Positive: {fp} | False Negative: {fn}')


def render_classification_report(y_true: pd.Series, y_pred: pd.Series) -> None:
    from sklearn.metrics import classification_report

    report = classification_report(y_true, y_pred, labels=[1, 0], target_names=['Yes', 'No'], output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).T.reset_index().rename(columns={'index': 'Class'})
    report_df = report_df[['Class', 'precision', 'recall', 'f1-score', 'support']]
    st.dataframe(report_df, use_container_width=True)


def render_model_comparison(models: dict[str, object], X: pd.DataFrame, y_true: pd.Series) -> None:
    rows = []
    for name, model in models.items():
        rows.append({'Model': name, **compute_metrics_for_model(model, X, y_true)})

    comparison_df = pd.DataFrame(rows)
    st.dataframe(comparison_df.round(4), use_container_width=True)

    melted = comparison_df.melt(id_vars='Model', value_vars=['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC'])
    fig = px.bar(melted, x='Model', y='value', color='variable', barmode='group', title='Model Metric Comparison')
    st.plotly_chart(fig, use_container_width=True)

    best_row = comparison_df.sort_values(['AUC', 'F1', 'Recall', 'MCC'], ascending=False).iloc[0]
    st.success(f'Best-performing model on the uploaded dataset: {best_row["Model"]} (AUC={best_row["AUC"]:.4f}, F1={best_row["F1"]:.4f})')


def main() -> None:
    st.set_page_config(page_title='Bank Marketing – Customer Subscription Prediction', layout='wide')
    st.title('Bank Marketing – Customer Subscription Prediction')
    st.caption('Machine learning dashboard for evaluating bank marketing subscription predictions.')

    st.sidebar.header('Navigation')
    page = st.sidebar.radio('Select a view', ['Single Model Evaluation', 'Model Comparison'])

    with st.sidebar:
        uploaded_file = st.file_uploader('Upload a CSV dataset for evaluation', type=['csv'])
        st.caption(
            'Supported: bank-additional CSVs (semicolon-delimited) and the processed dataset. '
            'The older bank-full/bank.csv files have a different feature schema and cannot be '
            'evaluated by these models.'
        )

    if not TEST_DATA_PATH.exists():
        st.error('The project test dataset is missing. Run train_models.py to generate test_data.csv and saved model artifacts.')
        return

    if uploaded_file is None:
        with st.container():
            st.info(
                'Upload a bank-additional CSV or processed dataset with the required features '
                'and target column y. Both yes/no and 1/0 target values are accepted.'
            )
        preview_df = pd.read_csv(TEST_DATA_PATH)
        st.subheader('Sample uploaded dataset preview')
        st.dataframe(preview_df.head(), use_container_width=True)
        return

    try:
        uploaded_df = read_uploaded_csv(uploaded_file)
        X_eval, y_true = validate_uploaded_data(uploaded_df)
    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.error(f'Invalid CSV: {exc}')
        return

    st.subheader('Uploaded dataset preview')
    st.dataframe(uploaded_df.head(10), use_container_width=True)

    try:
        models = load_all_models()
    except FileNotFoundError as exc:
        st.error(f'Model files are missing: {exc}. Run train_models.py to generate them before launching the app.')
        return

    if page == 'Single Model Evaluation':
        selected_name = st.selectbox('Choose a model', MODEL_OPTIONS)
        model = models[selected_name]
        y_pred = model.predict(X_eval)

        metrics = compute_metrics_for_model(model, X_eval, y_true)
        render_metric_cards(metrics)
        render_confusion_matrix(y_true, y_pred)
        render_classification_report(y_true, y_pred)

        pred_counts = pd.Series(y_pred).value_counts().sort_index()
        pred_df = pd.DataFrame({
            'Prediction': ['No', 'Yes'],
            'Count': [pred_counts.get(0, 0), pred_counts.get(1, 0)],
        })
        st.subheader('Subscription Prediction Distribution')
        st.bar_chart(pred_df.set_index('Prediction')['Count'])

    else:
        render_model_comparison(models, X_eval, y_true)


if __name__ == '__main__':
    main()
