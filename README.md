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
- Strongest ROC-AUC among the tested models.
- Balanced recall and campaign response detection, which is important for subscription prediction.
- Performs well in an imbalanced setting when using class weighting and a careful decision threshold.
- Lower precision than some alternatives, indicating that while it captures positive subscriptions effectively, it produces more false positives than the most conservative model.

### Decision Tree
- Captures non-linear relationships and interpretable decision rules.
- Competitive AUC and recall, but weaker precision and MCC than logistic regression.
- More sensitive to class imbalance and may overfit without careful tree depth control.

### K-Nearest Neighbors (KNN)
- Best raw accuracy because it classifies many majority-class examples correctly.
- However, it has the weakest AUC and poor recall in the positive class, which makes it less suitable for a minority-class business problem like term deposit acquisition.
- Its performance highlights how accuracy alone can be misleading in imbalanced classification tasks.

### Gaussian Naive Bayes
- Efficient and fast, with simple probabilistic assumptions.
- Performs reasonably for recall but lags behind other models in AUC and MCC.
- Assumes feature independence, which is a simplification for this marketing dataset and limits performance.

### Support Vector Machine (SVM)
- Achieved the highest F1 and MCC among the models, indicating robust discrimination in the positive class.
- Also showed the highest accuracy among the models.
- However, its ROC-AUC is lower than logistic regression, and it was computationally heavier to train.
- Still a strong option when balanced classification quality is prioritized.

## G. Overall Winner

The best overall model for this assignment is the Logistic Regression model because it delivered the strongest ROC-AUC (0.8009) and strong recall (0.6455), which are critical when evaluating a minority positive class in a highly imbalanced dataset. It also had a strong MCC and F1 score, making it the most balanced choice rather than the model with the highest raw accuracy alone.

The Support Vector Machine had the best F1 and MCC, but its AUC was lower and the assignment specifically prioritizes AUC, recall, and MCC in the model selection rationale. Therefore, Logistic Regression was selected as the overall winner for this project.

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

Deployment URL (placeholder): https://your-streamlit-app.streamlit.app/

## I. Instructions to Run Locally

1. Open a terminal in the project root.
2. Create and activate a virtual environment:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

4. Train the models and generate evaluation outputs:

```bash
python train_models.py
```

5. Run the Streamlit application:

```bash
streamlit run app.py
```

6. Open the local Streamlit URL displayed in the terminal.

## J. Project Structure

```text
ML-Assignment-2/
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
│   │           └── bank-additional.csv
│   └── processed/
│       ├── bank_marketing_processed.csv
│       └── eda_summary.txt
├── models/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── k_nearest_neighbors.joblib
│   ├── gaussian_naive_bayes.joblib
│   └── support_vector_machine.joblib
├── outputs/
│   ├── evaluation_results.csv
│   └── plots/
│       ├── accuracy_comparison.png
│       ├── auc_comparison.png
│       ├── f1_comparison.png
│       ├── mcc_comparison.png
│       ├── precision_comparison.png
│       ├── recall_comparison.png
│       ├── roc_comparison.png
│       ├── target_distribution.png
│       └── confusion_matrix and ROC plots for each model
├── notebooks/
├── screenshots/
├── src/
│   ├── __init__.py
│   ├── project_paths.py
│   ├── data_preprocessing.py
│   ├── model_training.py
│   └── model_evaluation.py
└── venv/
```

## Deployment Preparation for Streamlit Community Cloud

This project is organized to be compatible with Streamlit Community Cloud:

- All file paths are relative to the repository root.
- The trained model files are committed under `models/`.
- The application loads the saved models from the repository and does not train on the uploaded dataset.
- Missing model files are handled gracefully with a clear error message.
- The app can be launched from the repository root with `streamlit run app.py`.
- The dataset is not required at runtime for model inference; the app uses the uploaded CSV plus the saved model pipelines.

Important: this project is ready for deployment preparation, but no deployment claim is made unless a live Streamlit Cloud deployment has been created and verified by the user.

## Additional Notes

- The project uses reproducible preprocessing and a fixed random state.
- The evaluation is based on actual predictions from held-out data.
- `duration` is excluded intentionally to avoid leakage and to reflect realistic inference conditions.
- `unknown` categories are treated as missing and are handled via the preprocessing pipeline.
- The app expects the uploaded CSV to have the same feature columns as the test split, while the `y` column is used only for evaluation and is not passed as a model input.
- The `bank-additional-full.csv`, `bank-additional.csv`, processed dataset, and generated `test_data.csv` match the model feature schema. The separate older `bank-full.csv` and `bank.csv` files are from a different UCI dataset version with different features, so they cannot be evaluated by these trained models without retraining on that dataset.
