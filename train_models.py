from __future__ import annotations

from src.data_preprocessing import prepare_dataset, split_data
from src.model_evaluation import evaluate_all_models
from src.model_training import train_and_save_models


if __name__ == '__main__':
    df = prepare_dataset(save_processed=True)
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.2, random_state=42)
    pipelines = train_and_save_models(X_train, y_train, X_test, y_test)
    results_df = evaluate_all_models(pipelines, X_test, y_test)
    print('\nModel Evaluation Results:')
    print(results_df.round(4).to_string(index=False))
    best_model = results_df.sort_values(['AUC', 'F1', 'Recall', 'MCC'], ascending=False).iloc[0]
    print('\nBest model:')
    print(best_model.to_dict())
