from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.project_paths import MODELS_DIR, TEST_DATA_PATH

MODEL_OPTIONS = [
    'Logistic Regression',
    'Decision Tree',
    'K-Nearest Neighbors',
    'Gaussian Naive Bayes',
    'Support Vector Machine',
]
METRIC_NAMES = ['Accuracy', 'AUC', 'Precision', 'Recall', 'F1', 'MCC']


def slugify_model_name(name: str) -> str:
    return name.lower().replace(' ', '_').replace('-', '_')


def get_model_path(name: str) -> Path:
    return MODELS_DIR / f'{slugify_model_name(name)}.joblib'


def read_uploaded_csv(uploaded_file) -> pd.DataFrame:
    """Read comma- or semicolon-delimited CSVs and normalize their headers."""
    df = pd.read_csv(uploaded_file, sep=None, engine='python')
    df.columns = [str(column).strip().lstrip('\ufeff') for column in df.columns]
    return df


@st.cache_resource
def load_all_models() -> dict[str, object]:
    """Load saved pipelines once per app process; never fit models in the app."""
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
    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(f'The bundled test data was not found: {TEST_DATA_PATH}')

    expected_features = pd.read_csv(TEST_DATA_PATH, nrows=0).columns.drop('y').tolist()
    if 'y' not in df.columns:
        raise ValueError('The CSV must include the target column "y" for evaluation.')

    missing = [column for column in expected_features if column not in df.columns]
    if missing:
        raise ValueError(
            'This file does not match the dataset used to train these models. '
            f'Missing required feature columns: {missing}'
        )

    frame = df.copy()
    if 'duration' in frame.columns:
        frame = frame.drop(columns=['duration'])

    target_text = frame['y'].astype('string').str.strip().str.lower()
    target = target_text.map({'yes': 1, 'no': 0, '1': 1, '0': 0})
    if target.isna().any():
        invalid_values = target_text[target.isna()].drop_duplicates().head(5).tolist()
        raise ValueError(
            'The "y" column must contain only yes/no or 1/0 values. '
            f'Invalid value(s): {invalid_values}'
        )

    features = frame[expected_features].copy()
    for column in features.select_dtypes(include=['object', 'string']).columns:
        features[column] = features[column].replace('unknown', pd.NA)
    return features, target.astype(int)


def predict_and_score(model, X: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return labels and continuous scores for the positive (yes) class."""
    predictions = pd.Series(model.predict(X), index=X.index, dtype='int64')
    if hasattr(model, 'predict_proba'):
        scores = pd.Series(model.predict_proba(X)[:, 1], index=X.index)
    else:
        scores = pd.Series(model.decision_function(X), index=X.index)
    return predictions, scores


def compute_metrics(y_true: pd.Series, y_pred: pd.Series, y_score: pd.Series) -> dict:
    auc = roc_auc_score(y_true, y_score) if y_true.nunique() == 2 else None
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'AUC': auc,
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'MCC': matthews_corrcoef(y_true, y_pred),
    }


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1440px;}
        [data-testid="stMetric"] {
            background: linear-gradient(135deg, rgba(33, 150, 243, .10), rgba(33, 150, 243, .025));
            border: 1px solid rgba(33, 150, 243, .17);
            padding: 1rem 1.1rem;
            border-radius: 14px;
        }
        [data-testid="stMetricLabel"] p {font-size: .82rem; font-weight: 600;}
        .app-kicker {color: #3182ce; text-transform: uppercase; letter-spacing: .12em;
                     font-size: .75rem; font-weight: 700; margin-bottom: .35rem;}
        .app-subtitle {color: #64748b; font-size: 1.02rem; margin-top: -.25rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> tuple[object, bool, str, bool, bool, int]:
    st.sidebar.markdown('## 1 · Test data')
    st.sidebar.caption('Upload a labelled CSV with the required model features and target column `y`.')
    uploaded_file = st.sidebar.file_uploader(
        'Upload a CSV dataset for evaluation',
        type=['csv'],
        help='Comma- and semicolon-delimited bank-additional CSV files are supported.',
        label_visibility='collapsed',
    )
    use_bundled = st.sidebar.toggle(
        'No file? Use the bundled test_data.csv',
        value=False,
        help='Use the held-out test split included with this project.',
    )
    if TEST_DATA_PATH.exists():
        st.sidebar.download_button(
            'Download test_data.csv',
            data=TEST_DATA_PATH.read_bytes(),
            file_name='test_data.csv',
            mime='text/csv',
            width='stretch',
        )

    st.sidebar.markdown('## 2 · Model')
    selected_model = st.sidebar.selectbox('Choose a classifier', MODEL_OPTIONS)

    st.sidebar.markdown('## 3 · Display')
    show_roc = st.sidebar.checkbox('Show ROC curve', value=True)
    show_preview = st.sidebar.checkbox('Show data preview', value=True)
    preview_rows = st.sidebar.select_slider(
        'Preview rows',
        options=[10, 25, 50, 100],
        value=10,
        disabled=not show_preview,
    )

    with st.sidebar.expander('Expected columns'):
        if TEST_DATA_PATH.exists():
            expected = pd.read_csv(TEST_DATA_PATH, nrows=0).columns.tolist()
            st.caption('The original `duration` field is accepted but ignored (leakage prevention).')
            st.code(', '.join(expected), language=None)
        st.caption(
            'The older `bank-full.csv` and `bank.csv` variants use a different feature schema '
            'and are not compatible with these trained models.'
        )

    st.sidebar.markdown('---')
    st.sidebar.caption('Models are loaded from saved pipelines. Uploaded data is never used for training.')
    return uploaded_file, use_bundled, selected_model, show_roc, show_preview, preview_rows


def render_confusion_matrix(y_true: pd.Series, y_pred: pd.Series) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=[1, 0])
    matrix_df = pd.DataFrame(
        matrix,
        index=['Actual Yes', 'Actual No'],
        columns=['Predicted Yes', 'Predicted No'],
    )
    fig = px.imshow(
        matrix_df,
        text_auto=True,
        color_continuous_scale='Blues',
        aspect='auto',
        labels={'x': 'Prediction', 'y': 'Actual class', 'color': 'Records'},
    )
    fig.update_layout(height=390, margin=dict(l=10, r=10, t=35, b=10))
    st.plotly_chart(fig, width='stretch')
    tp, fn, fp, tn = matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1]
    st.caption(f'True positive: {tp:,}  ·  False negative: {fn:,}  ·  False positive: {fp:,}  ·  True negative: {tn:,}')


def render_roc_curve(y_true: pd.Series, y_score: pd.Series, auc: float | None) -> None:
    if auc is None:
        st.info('A ROC-AUC curve requires both target classes in the uploaded data.')
        return
    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, y_score)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=false_positive_rate,
        y=true_positive_rate,
        mode='lines',
        name=f'Model (AUC {auc:.3f})',
        line=dict(color='#2383e2', width=3),
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode='lines', name='Random baseline',
        line=dict(color='#94a3b8', dash='dash'),
    ))
    fig.update_layout(
        title='Receiver Operating Characteristic',
        xaxis_title='False positive rate',
        yaxis_title='True positive rate',
        height=390,
        margin=dict(l=10, r=10, t=55, b=10),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
    )
    st.plotly_chart(fig, width='stretch')


def render_classification_report(y_true: pd.Series, y_pred: pd.Series) -> None:
    report = classification_report(
        y_true,
        y_pred,
        labels=[1, 0],
        target_names=['Yes', 'No'],
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).T.reset_index().rename(columns={'index': 'Class'})
    st.dataframe(
        report_df[['Class', 'precision', 'recall', 'f1-score', 'support']].round(4),
        width='stretch',
        hide_index=True,
    )


def render_comparison(models: dict[str, object], X: pd.DataFrame, y_true: pd.Series) -> None:
    with st.spinner('Scoring the uploaded rows with all saved models…'):
        rows = []
        for name, model in models.items():
            predictions, scores = predict_and_score(model, X)
            rows.append({'Model': name, **compute_metrics(y_true, predictions, scores)})
    comparison = pd.DataFrame(rows)
    st.dataframe(comparison.round(4), width='stretch', hide_index=True)
    chart_data = comparison.melt(id_vars='Model', value_vars=METRIC_NAMES, var_name='Metric', value_name='Score')
    chart = px.bar(
        chart_data,
        x='Model',
        y='Score',
        color='Metric',
        barmode='group',
        title='Model comparison',
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    chart.update_layout(height=420, margin=dict(l=10, r=10, t=60, b=10))
    st.plotly_chart(chart, width='stretch')
    valid_auc = comparison.dropna(subset=['AUC'])
    if not valid_auc.empty:
        best = valid_auc.sort_values(['AUC', 'F1', 'Recall', 'MCC'], ascending=False).iloc[0]
        st.success(
            f'Highest-ranked model for this dataset: **{best["Model"]}** '
            f'(AUC {best["AUC"]:.3f}, F1 {best["F1"]:.3f}).'
        )


def main() -> None:
    st.set_page_config(
        page_title='ML Model Evaluation Dashboard',
        page_icon='📊',
        layout='wide',
        initial_sidebar_state='expanded',
    )
    inject_styles()
    uploaded_file, use_bundled, selected_model, show_roc, show_preview, preview_rows = render_sidebar()

    st.markdown('<div class="app-kicker">Machine learning · classification</div>', unsafe_allow_html=True)
    st.title('Model Evaluation Dashboard')
    st.markdown(
        '<p class="app-subtitle">Evaluate saved classifiers on labelled test data — '
        'without retraining or fitting anything to your upload.</p>',
        unsafe_allow_html=True,
    )

    if not uploaded_file and not use_bundled:
        st.info('Upload a test CSV in the sidebar, or turn on **Use the bundled test_data.csv** to try the dashboard.')
        return

    try:
        if uploaded_file is not None and not use_bundled:
            uploaded_file.seek(0)
            evaluation_df = read_uploaded_csv(uploaded_file)
            data_source = uploaded_file.name
        else:
            evaluation_df = pd.read_csv(TEST_DATA_PATH)
            data_source = 'Bundled held-out test_data.csv'
        X_eval, y_true = validate_uploaded_data(evaluation_df)
    except FileNotFoundError as exc:
        st.error(f'Required project file is missing: {exc}')
        return
    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.error(f'Could not use this CSV: {exc}')
        return

    try:
        models = load_all_models()
    except FileNotFoundError as exc:
        st.error(f'Saved model files are missing: {exc}')
        return

    st.caption(f'**Data source:** {data_source}  ·  **Rows:** {len(evaluation_df):,}  ·  **Model:** {selected_model}')
    if show_preview:
        with st.expander(f'Data preview · first {min(preview_rows, len(evaluation_df)):,} of {len(evaluation_df):,} rows', expanded=False):
            st.dataframe(
                evaluation_df.head(preview_rows),
                width='stretch',
                height=360,
                hide_index=True,
            )

    st.markdown('### Evaluation')
    selected_model_obj = models[selected_model]
    try:
        with st.spinner(f'Evaluating {selected_model} on {len(X_eval):,} rows…'):
            y_pred, y_score = predict_and_score(selected_model_obj, X_eval)
            metrics = compute_metrics(y_true, y_pred, y_score)
    except (ValueError, TypeError, KeyError) as exc:
        st.error(f'The selected model could not evaluate this dataset: {exc}')
        return

    st.markdown('#### Performance metrics')
    metric_cols = st.columns(6)
    for column, metric_name in zip(metric_cols, METRIC_NAMES):
        value = metrics[metric_name]
        column.metric(metric_name, 'N/A' if value is None else f'{value:.4f}')
    if metrics['AUC'] is None:
        st.warning('AUC is unavailable because the uploaded target contains only one class.')

    counts = pd.Series(y_pred).value_counts().reindex([0, 1], fill_value=0)
    subscription_rate = float(counts.loc[1] / max(len(y_pred), 1))
    summary_cols = st.columns(3)
    summary_cols[0].metric('Test records', f'{len(y_pred):,}')
    summary_cols[1].metric('Predicted subscriptions', f'{counts.loc[1]:,}')
    summary_cols[2].metric('Predicted subscription rate', f'{subscription_rate:.1%}')

    tab_performance, tab_confusion, tab_report, tab_compare = st.tabs(
        ['Performance', 'Confusion matrix', 'Classification report', 'Compare models']
    )
    with tab_performance:
        if show_roc:
            render_roc_curve(y_true, y_score, metrics['AUC'])
        prediction_df = pd.DataFrame({
            'Prediction': ['No subscription', 'Subscription'],
            'Records': [int(counts.loc[0]), int(counts.loc[1])],
        })
        prediction_chart = px.bar(
            prediction_df,
            x='Prediction',
            y='Records',
            color='Prediction',
            color_discrete_map={'No subscription': '#93c5fd', 'Subscription': '#2563eb'},
            title='Predicted class distribution',
        )
        prediction_chart.update_layout(showlegend=False, height=350, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(prediction_chart, width='stretch')
    with tab_confusion:
        render_confusion_matrix(y_true, y_pred)
    with tab_report:
        render_classification_report(y_true, y_pred)
    with tab_compare:
        if st.button('Compare all five saved models', type='primary'):
            render_comparison(models, X_eval, y_true)
        else:
            st.caption('Run a side-by-side comparison on the same uploaded rows. This can take longer for SVM.')

    with st.expander('What are these models?', expanded=False):
        st.write(
            'The dashboard compares Logistic Regression, Decision Tree, K-Nearest Neighbors, '
            'Gaussian Naive Bayes, and Support Vector Machine. Each saved artifact contains its '
            'training-fitted preprocessing steps and classifier. The target column is used only '
            'to calculate evaluation metrics; it is never passed into a model.'
        )


if __name__ == '__main__':
    main()
