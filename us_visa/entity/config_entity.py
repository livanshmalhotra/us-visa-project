import os
from datetime import datetime
from us_visa.constants import *

class TrainingPipelineConfig:
    def __init__(self):
        self.pipeline_name: str = PIPELINE_NAME
        self.artifact_dir: str = os.path.join(ARTIFACT_DIR, datetime.now().strftime("%m_%d_%Y_%H_%M_%S"))

class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_ingestion_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME
        )
        self.feature_store_file_path: str = os.path.join(
            self.data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, DATA_INGESTION_RAW_DATA_FILE_NAME
        )
        self.train_file_path: str = os.path.join(
            self.data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, DATA_INGESTION_TRAIN_FILE_NAME
        )
        self.test_file_path: str = os.path.join(
            self.data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, DATA_INGESTION_TEST_FILE_NAME
        )
        self.train_test_split_ratio: float = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
        self.collection_name: str = COLLECTION_NAME

class DataValidationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_validation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, DATA_VALIDATION_DIR_NAME
        )
        self.schema_file_path: str = SCHEMA_FILE_PATH
        self.report_file_path: str = os.path.join(
            self.data_validation_dir, DATA_VALIDATION_REPORT_FILE_NAME
        )
        self.drift_report_file_path: str = os.path.join(
            self.data_validation_dir, DATA_VALIDATION_DRIFT_REPORT_FILE_NAME
        )

class DataTransformationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_transformation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, DATA_TRANSFORMATION_DIR_NAME
        )
        self.transformed_train_file_path: str = os.path.join(
            self.data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DIR, DATA_INGESTION_TRAIN_FILE_NAME.replace("csv", "npy")
        )
        self.transformed_test_file_path: str = os.path.join(
            self.data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_DIR, DATA_INGESTION_TEST_FILE_NAME.replace("csv", "npy")
        )
        self.transformed_object_file_path: str = os.path.join(
            self.data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR, PREPROCESSING_OBJECT_FILE_NAME
        )

class ModelTrainerConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.model_trainer_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, MODEL_TRAINER_DIR_NAME
        )
        self.trained_model_file_path: str = os.path.join(
            self.model_trainer_dir, MODEL_TRAINER_TRAINED_MODEL_DIR, MODEL_FILE_NAME
        )
        self.expected_accuracy: float = MODEL_TRAINER_EXPECTED_SCORE
        self.overfitting_limit: float = MODEL_TRAINER_OVERFITTING_UNDERFITTING_THRESHOLD

class ModelEvaluationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.model_evaluation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, MODEL_EVALUATION_DIR_NAME
        )
        self.report_file_path: str = os.path.join(
            self.model_evaluation_dir, MODEL_EVALUATION_REPORT_FILE_NAME
        )
        self.changed_threshold_score: float = MODEL_EVALUATION_CHANGED_THRESHOLD_SCORE

class ModelPusherConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.model_pusher_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, MODEL_PUSHER_DIR_NAME
        )
        self.saved_model_dir: str = MODEL_PUSHER_SAVED_MODEL_DIR
        self.saved_model_file_path: str = os.path.join(
            self.saved_model_dir, MODEL_FILE_NAME
        )
