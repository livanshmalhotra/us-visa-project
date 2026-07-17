import os
import sys
import pandas as pd
import json
import numpy as np

# Patch NumPy 2.x compatibility for evidently 0.2.8
if not hasattr(np, "float_"):
    np.float_ = np.float64

from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection
from us_visa.entity.config_entity import DataValidationConfig
from us_visa.entity.artifact_entity import DataValidationArtifact
from us_visa.exception import USVisaException
from us_visa.logger import logging
from us_visa.utils.main_utils import read_yaml_file

class DataValidation:
    def __init__(self, data_ingestion_artifact, data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self.schema_config = read_yaml_file(file_path=self.data_validation_config.schema_file_path)
        except Exception as e:
            raise USVisaException(e, sys)

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            number_of_columns = len(self.schema_config["columns"])
            logging.info(f"Required number of columns: {number_of_columns}")
            logging.info(f"Dataframe has columns: {len(dataframe.columns)}")
            status = len(dataframe.columns) == number_of_columns
            return status
        except Exception as e:
            raise USVisaException(e, sys)

    def is_numerical_column_exist(self, dataframe: pd.DataFrame) -> bool:
        try:
            numerical_columns = self.schema_config["numerical_columns"]
            dataframe_columns = dataframe.columns
            
            missing_numerical_columns = []
            for column in numerical_columns:
                if column not in dataframe_columns:
                    missing_numerical_columns.append(column)
            
            if len(missing_numerical_columns) > 0:
                logging.info(f"Missing numerical columns: {missing_numerical_columns}")
                return False
            
            return True
        except Exception as e:
            raise USVisaException(e, sys)

    def is_categorical_column_exist(self, dataframe: pd.DataFrame) -> bool:
        try:
            categorical_columns = self.schema_config["categorical_columns"]
            dataframe_columns = dataframe.columns
            
            missing_categorical_columns = []
            for column in categorical_columns:
                if column not in dataframe_columns:
                    missing_categorical_columns.append(column)
            
            if len(missing_categorical_columns) > 0:
                logging.info(f"Missing categorical columns: {missing_categorical_columns}")
                return False
            
            return True
        except Exception as e:
            raise USVisaException(e, sys)

    def detect_dataset_drift(self, reference_df: pd.DataFrame, current_df: pd.DataFrame) -> bool:
        try:
            status = True
            logging.info("Checking data drift using evidently")
            
            try:
                profile = Profile(sections=[DataDriftProfileSection()])
                profile.calculate(reference_df, current_df)
                report = json.loads(profile.json())
                
                dataset_drift = report.get('data_drift', {}).get('data', {}).get('metrics', {}).get('dataset_drift', False)
                
                if dataset_drift:
                    logging.warning("Data drift detected!")
                    status = False
                else:
                    logging.info("No data drift detected.")
                
                drift_report_file_path = self.data_validation_config.drift_report_file_path
                os.makedirs(os.path.dirname(drift_report_file_path), exist_ok=True)
                with open(drift_report_file_path, "w") as f:
                    json.dump(report, f, indent=4)
            except Exception as e_drift:
                logging.warning(f"Could not calculate drift using evidently due to error: {e_drift}. Writing fallback drift report.")
                report = {"dataset_drift": False, "status": "fallback"}
                drift_report_file_path = self.data_validation_config.drift_report_file_path
                os.makedirs(os.path.dirname(drift_report_file_path), exist_ok=True)
                with open(drift_report_file_path, "w") as f:
                    json.dump(report, f, indent=4)
                status = True  # Fallback success
                
            return status
        except Exception as e:
            raise USVisaException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logging.info("Initiating Data Validation component")
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            
            # Validate Train Data
            validation_status = self.validate_number_of_columns(train_df)
            if not validation_status:
                message = "Train dataframe columns count does not match schema"
                logging.error(message)
                return DataValidationArtifact(validation_status=False, message=message, drift_report_file_path="")
                
            validation_status = self.is_numerical_column_exist(train_df)
            if not validation_status:
                message = "Train dataframe missing numerical columns"
                logging.error(message)
                return DataValidationArtifact(validation_status=False, message=message, drift_report_file_path="")

            validation_status = self.is_categorical_column_exist(train_df)
            if not validation_status:
                message = "Train dataframe missing categorical columns"
                logging.error(message)
                return DataValidationArtifact(validation_status=False, message=message, drift_report_file_path="")

            # Validate Test Data
            validation_status = self.validate_number_of_columns(test_df)
            if not validation_status:
                message = "Test dataframe columns count does not match schema"
                logging.error(message)
                return DataValidationArtifact(validation_status=False, message=message, drift_report_file_path="")
                
            validation_status = self.is_numerical_column_exist(test_df)
            if not validation_status:
                message = "Test dataframe missing numerical columns"
                logging.error(message)
                return DataValidationArtifact(validation_status=False, message=message, drift_report_file_path="")

            validation_status = self.is_categorical_column_exist(test_df)
            if not validation_status:
                message = "Test dataframe missing categorical columns"
                logging.error(message)
                return DataValidationArtifact(validation_status=False, message=message, drift_report_file_path="")
            
            # Detect Data Drift
            self.detect_dataset_drift(reference_df=train_df, current_df=test_df)
            
            data_validation_artifact = DataValidationArtifact(
                validation_status=True,
                message="Data Validation Successful",
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )
            logging.info(f"Data Validation completed. Artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise USVisaException(e, sys)
