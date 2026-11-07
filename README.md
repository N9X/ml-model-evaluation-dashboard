# Bank Marketing – Customer Subscription Prediction

## A. Problem Statement

This project addresses the business problem of predicting whether a banking customer will subscribe to a term deposit after a direct marketing campaign. The dataset comes from the UCI Bank Marketing dataset and is a classic binary classification task. The objective is to use customer demographics, financial indicators, and campaign history to identify likely subscribers and support more targeted, efficient marketing strategies.

The target variable is `y`, where `yes` indicates a successful subscription and `no` indicates no subscription. The project builds, evaluates, and compares multiple machine learning classifiers, saves the trained models, and exposes the workflow through a professional Streamlit dashboard.

## B. Dataset Description

- Dataset source: UCI Machine Learning Repository, Bank Marketing dataset
- Dataset file: `dataset/raw/bank-additional/bank-additional/bank-additional-full.csv`
- Dataset size: 41,188 records
- Features before leakage exclusion: 20 input features
- Target variable: `y` with classes `yes` and `no`
- Data format: semicolon-delimited CSV (`;`)

### Feature overview

The dataset includes demographic and campaign-related characteristics such as:

- `age`: customer age
- `job`: occupation category
- `marital`: marital status
- `education`: educational level
- `default`: whether the customer has credit in default
- `housing`: housing loan status
- `loan`: personal loan status
- `contact`: communication type used in the campaign
- `month`: last contact month
- `day_of_week`: last contact day
- `campaign`: number of contacts during this campaign
- `pdays`: days since previous contact
- `previous`: number of contacts before this campaign
- `poutcome`: outcome of previous marketing campaign
- `emp.var.rate`, `cons.price.idx`, `cons.conf.idx`, `euribor3m`, `nr.employed`: macroeconomic context features

### Preprocessing decisions

- The column `duration` was intentionally excluded from predictive features because it is only known after the call and would create data leakage.
- `unknown` values in categorical features were converted to missing values and then handled through preprocessing using `SimpleImputer` and `OneHotEncoder`.
- The raw dataset was preserved and a processed copy was stored in `dataset/processed/`.
- The target variable was encoded to binary integers: `yes = 1`, `no = 0`.
- A stratified 80:20 train-test split was used with `random_state=42` for reproducibility.
- Preprocessing was fitted only on the training split to avoid leakage.

## C. GitHub Repository Link

GitHub repository URL: https://github.com/N9X/ml-model-evaluation-dashboard

## D. Models Used

The following five supervised classification algorithms were trained and evaluated:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

The models all used the same train-test split and the same target encoding. Each trained model was saved as a complete preprocessing + classifier pipeline using Joblib.

## E. Model Evaluation Comparison

The actual evaluation results generated from the held-out test dataset are shown below.

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8352 | 0.8009 | 0.3679 | 0.6455 | 0.4687 | 0.4007 |
| Decision Tree | 0.8178 | 0.7958 | 0.3432 | 0.6756 | 0.4552 | 0.3893 |
| Support Vector Machine | 0.8562 | 0.7798 | 0.4098 | 0.6293 | 0.4964 | 0.4299 |
| Gaussian Naive Bayes | 0.8049 | 0.7755 | 0.3172 | 0.6347 | 0.4230 | 0.3490 |
| K-Nearest Neighbors | 0.8951 | 0.7559 | 0.5731 | 0.2705 | 0.3675 | 0.3451 |

## F. Model Performance Observations

### Logistic Regression
- Highest ROC-AUC (0.8009), making it the top model under the project's AUC-first ranking.
- Recall is 0.6455, second only to the Decision Tree's 0.6756.
- Accuracy is 0.8352, with precision 0.3679, F1 0.4687, and MCC 0.4007.
- Its class-weighted configuration offers useful positive-class detection, though its precision shows that false positives remain a trade-off.

### Decision Tree
- Captures non-linear relationships and can express interpretable decision rules.
- Has the second-highest AUC (0.7958) and the highest recall (0.6756).
- Accuracy is 0.8178, precision 0.3432, F1 0.4552, and MCC 0.3893; its recall advantage comes with relatively low precision.

### K-Nearest Neighbors (KNN)
- Highest accuracy (0.8951) and precision (0.5731) among the five models.
- It has the lowest AUC (0.7559), recall (0.2705), F1 (0.3675), and MCC (0.3451).
- The low recall indicates that it misses many actual subscribers; its high accuracy is therefore not sufficient evidence of good minority-class performance.

### Gaussian Naive Bayes
- Accuracy is 0.8049, AUC 0.7755, precision 0.3172, recall 0.6347, F1 0.4230, and MCC 0.3490.
- Its recall is competitive, but its precision and overall discrimination are weaker than the top-ranked alternatives.
- The conditional-independence assumption makes this a simple probabilistic baseline, though it may not capture dependencies among these features.

### Support Vector Machine (SVM)
- Highest F1 (0.4964) and MCC (0.4299), and second-highest accuracy (0.8562).
- Its AUC is 0.7798, behind Logistic Regression (0.8009) and Decision Tree (0.7958); precision is 0.4098 and recall is 0.6293.
- These results make SVM a strong choice when F1 and MCC are emphasized, although its training was computationally heavier in this environment.

## G. Overall Winner

The project ranks models AUC first, followed by F1, recall, and MCC as tie-breakers. Logistic Regression is therefore the overall winner because it has the highest AUC (0.8009); no later metric displaces it under this ordering. Its recall is also strong at 0.6455, while its F1 (0.4687) and MCC (0.4007) are competitive but not the highest.

This is an AUC-first selection, not a claim that Logistic Regression leads every metric. SVM has the best F1 (0.4964) and MCC (0.4299), while KNN has the best accuracy (0.8951) and precision (0.5731). Logistic Regression is selected because the ranking prioritizes AUC, and its ROC-AUC is the highest.

## H. Streamlit Application

The project contains an interactive Streamlit evaluation dashboard in `app.py` titled:

`Model Evaluation Dashboard`

The application allows users to:

- Upload a CSV dataset for validation
- Optionally try the bundled held-out test data without uploading a file
- Read comma- or semicolon-delimited CSVs, including the original semicolon-delimited UCI files
- Evaluate targets encoded as either `yes`/`no` or `1`/`0`
- Confirm that required columns and target values are present
- Select a saved model from the five trained pipelines
- View the uploaded dataset preview and row count
- View key metrics, ROC curve, confusion matrix, class report, and predicted class distribution
- Compare all five models side by side on demand

The app is designed to load the saved joblib models from the `models/` directory and never retrain on uploaded data.

Deployment URL: [https://ml-model-evaluation-dashboard-c6xry4pdpoh9dcapfefetw.streamlit.app/](https://ml-model-evaluation-dashboard-c6xry4pdpoh9dcapfefetw.streamlit.app/)

## I. Instructions to Run Locally

1. Clone the repository and enter the project directory:

```bash
git clone https://github.com/N9X/ml-model-evaluation-dashboard.git
cd ml-model-evaluation-dashboard
```

2. Create and activate a virtual environment:

```bash
python -m venv venv
# Windows PowerShell
.\venv\Scripts\Activate.ps1
# macOS/Linux
source venv/bin/activate
```

3. Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

4. Start the Streamlit application using the included trained models:

```bash
streamlit run app.py
```

5. In the sidebar, upload a compatible labelled CSV, or switch on **No file? Use the bundled test_data.csv** to try the included held-out test data.

The trained Joblib models, evaluation results, and `test_data.csv` are included in the repository, so retraining is not needed to run the application. To reproduce the training and regenerate the models and evaluation outputs instead, run:

```bash
python train_models.py
```

After launching, open the local Streamlit URL displayed in the terminal.

## J. Project Structure

```text
ml-model-evaluation-dashboard/
├── .devcontainer/
│   └── devcontainer.json
├── app.py
├── train_models.py
├── inspect_dataset.py
├── requirements.txt
├── README.md
├── .gitignore
├── test_data.csv
├── dataset/
│   ├── raw/
│   │   └── bank-additional/
│   │       └── bank-additional/
│   │           ├── bank-additional-full.csv
│   │           ├── bank-additional-names.txt
│   │           └── bank-additional.csv
│   └── processed/
│       ├── bank_marketing_processed.csv
│       └── eda_summary.txt
├── models/
│   ├── decision_tree.joblib
│   ├── gaussian_naive_bayes.joblib
│   ├── k_nearest_neighbors.joblib
│   ├── logistic_regression.joblib
│   └── support_vector_machine.joblib
├── outputs/
│   ├── evaluation_results.csv
│   └── plots/
│       ├── accuracy_comparison.png
│       ├── auc_comparison.png
│       ├── decision_tree_confusion_matrix.png
│       ├── decision_tree_roc_curve.png
│       ├── f1_comparison.png
│       ├── gaussian_naive_bayes_confusion_matrix.png
│       ├── gaussian_naive_bayes_roc_curve.png
│       ├── k_nearest_neighbors_confusion_matrix.png
│       ├── k_nearest_neighbors_roc_curve.png
│       ├── logistic_regression_confusion_matrix.png
│       ├── logistic_regression_roc_curve.png
│       ├── mcc_comparison.png
│       ├── missing_values.png
│       ├── precision_comparison.png
│       ├── recall_comparison.png
│       ├── roc_comparison.png
│       ├── support_vector_machine_confusion_matrix.png
│       ├── support_vector_machine_roc_curve.png
│       └── target_distribution.png
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── model_evaluation.py
│   ├── model_training.py
│   └── project_paths.py
```

## Deployment Preparation for Streamlit Community Cloud

This project is organized to be compatible with Streamlit Community Cloud:

- All file paths are relative to the repository root.
- The trained model files are committed under `models/`.
- The application loads the saved models from the repository and does not train on the uploaded dataset.
- Missing model files are handled gracefully with a clear error message.
- The app can be launched from the repository root with `streamlit run app.py`.
- The dataset is not required at runtime for model inference; the app uses the uploaded CSV plus the saved model pipelines.

The application is deployed at [the Streamlit Community Cloud URL](https://ml-model-evaluation-dashboard-c6xry4pdpoh9dcapfefetw.streamlit.app/).

The `screenshots/` directory is not included in the repository; add any required BITS Virtual Lab screenshot manually after capturing it. No BITS Virtual Lab execution or screenshot is claimed here.

## Additional Notes

- The project uses reproducible preprocessing and a fixed random state.
- The evaluation is based on actual predictions from held-out data.
- `duration` is excluded intentionally to avoid leakage and to reflect realistic inference conditions.
- `unknown` categories are treated as missing and are handled via the preprocessing pipeline.
- The app expects the uploaded CSV to have the same feature columns as the test split, while the `y` column is used only for evaluation and is not passed as a model input.
- The `bank-additional-full.csv`, `bank-additional.csv`, processed dataset, and generated `test_data.csv` match the model feature schema. The separate older `bank-full.csv` and `bank.csv` files are from a different UCI dataset version with different features, so they cannot be evaluated by these trained models without retraining on that dataset.
