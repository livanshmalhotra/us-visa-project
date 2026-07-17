import os

# Database constants
DATABASE_NAME = "US_VISA"
COLLECTION_NAME = "visa_data"
MONGODB_URL_KEY = "MONGODB_URL"

# Pipeline constants
PIPELINE_NAME = "us_visa"
ARTIFACT_DIR = "artifact"
MODEL_FILE_NAME = "model.pkl"
PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")

# Data Ingestion constants
DATA_INGESTION_DIR_NAME = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR = "feature_store"
DATA_INGESTION_INGESTED_DIR = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO = 0.2
DATA_INGESTION_RAW_DATA_FILE_NAME = "easyvisa.csv"
DATA_INGESTION_TRAIN_FILE_NAME = "train.csv"
DATA_INGESTION_TEST_FILE_NAME = "test.csv"

# Data Validation constants
DATA_VALIDATION_DIR_NAME = "data_validation"
DATA_VALIDATION_REPORT_FILE_NAME = "report.yaml"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME = "drift_report.json"

# Data Transformation constants
DATA_TRANSFORMATION_DIR_NAME = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DIR = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR = "transformed_object"

# Model Trainer constants
MODEL_TRAINER_DIR_NAME = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR = "trained_model"
MODEL_TRAINER_EXPECTED_SCORE = 0.6
MODEL_TRAINER_OVERFITTING_UNDERFITTING_THRESHOLD = 0.1

# Model Evaluation constants
MODEL_EVALUATION_DIR_NAME = "model_evaluation"
MODEL_EVALUATION_REPORT_FILE_NAME = "report.yaml"
MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE = 0.02

# Model Pusher constants
MODEL_PUSHER_DIR_NAME = "model_pusher"
MODEL_PUSHER_SAVED_MODEL_DIR = "models"
