from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = ROOT_DIR / "dataset"
RAW_DATA_PATH = DATASET_DIR / "raw" / "bank-additional" / "bank-additional" / "bank-additional-full.csv"
PROCESSED_DIR = DATASET_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
OUTPUTS_DIR = ROOT_DIR / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
TEST_DATA_PATH = ROOT_DIR / "test_data.csv"
REPORTS_DIR = ROOT_DIR / "reports"
