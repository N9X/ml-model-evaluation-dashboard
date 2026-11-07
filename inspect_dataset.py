from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data_preprocessing import prepare_dataset
from src.project_paths import OUTPUTS_DIR, PLOTS_DIR, PROCESSED_DIR


def inspect_dataset() -> pd.DataFrame:
    df = prepare_dataset(save_processed=True)

    print('Dataset shape:', df.shape)
    print('\nColumns:', list(df.columns))
    print('\nData types:\n', df.dtypes)
    print('\nSample records:\n', df.head(5).to_string(index=False))
    print('\nMissing values:\n', df.isna().sum().sort_values(ascending=False).head(10))
    print('\nDuplicate record count:', df.duplicated().sum())

    categorical_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    print('\nCategorical columns:', categorical_cols)
    print('\nNumeric columns:', numeric_cols)
    print('\nTarget distribution:\n', df['y'].value_counts().to_string())
    print('\nClass imbalance ratio:', df['y'].value_counts(normalize=True).to_dict())

    for col in categorical_cols:
        if 'unknown' in str(df[col].dropna().unique()):
            print(f"\nUnknown values in {col}: {df[col].isin(['unknown']).sum()}")

    target_plot_path = PLOTS_DIR / 'target_distribution.png'
    target_plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    target_counts = df['y'].value_counts().sort_index()
    target_counts.index = ['No', 'Yes']
    ax.bar(target_counts.index, target_counts.values, color=['#1f77b4', '#ff7f0e'])
    ax.set_title('Target Distribution (y)')
    ax.set_xlabel('Target Class')
    ax.set_ylabel('Count')
    fig.tight_layout()
    fig.savefig(target_plot_path, dpi=200)
    plt.close(fig)

    missing_plot_path = PLOTS_DIR / 'missing_values.png'
    missing_data = df.isna().sum().sort_values(ascending=False)
    missing_data = missing_data[missing_data > 0]
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.barplot(x=missing_data.index, y=missing_data.values, hue=missing_data.index, dodge=False, palette='viridis', ax=ax)
    if ax.legend_ is not None:
        ax.legend_.remove()
    ax.set_title('Missing Values by Feature')
    ax.set_xlabel('Feature')
    ax.set_ylabel('Missing Count')
    ax.tick_params(axis='x', rotation=35)
    fig.tight_layout()
    fig.savefig(missing_plot_path, dpi=200)
    plt.close(fig)

    summary_path = PROCESSED_DIR / 'eda_summary.txt'
    summary_path.write_text(
        f"Shape: {df.shape}\n"
        f"Duplicate rows: {df.duplicated().sum()}\n"
        f"Target distribution: {df['y'].value_counts().to_dict()}\n"
        f"Unknown categorical values handled by converting them to NaN and imputing as 'missing' during preprocessing.\n"
        f"Duration excluded from predictive features because it is only known after the call and would create data leakage.\n",
        encoding='utf-8',
    )

    return df


if __name__ == '__main__':
    inspect_dataset()
    print('\nEDA output saved to', OUTPUTS_DIR)
    print('Processed data saved to', PROCESSED_DIR)
