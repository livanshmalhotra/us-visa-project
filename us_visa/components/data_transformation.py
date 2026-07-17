import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from us_visa.entity.config_entity import DataTransformationConfig
from us_visa.entity.artifact_entity import DataTransformationArtifact
from us_visa.exception import USVisaException
from us_visa.logger import logging
from us_visa.utils.main_utils import read_yaml_file, save_numpy_array_data, save_object
from us_visa.constants import SCHEMA_FILE_PATH, PREPROCESSING_OBJECT_FILE_NAME

class DataTransformation:
    def __init__(self, data_ingestion_artifact, data_transformation_config: DataTransformationConfig, data_validation_artifact):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self.schema_config = read_yaml_file(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise USVisaException(e, sys)

    def get_data_transformer_object(self) -> ColumnTransformer:
        """
        Method Name : get_data_transformer_object
        Description : Creates and returns a ColumnTransformer preprocessing object
        """
        try:
            logging.info("Creating preprocessing pipeline")
            numeric_transformer = StandardScaler()
            oh_transformer = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            
            num_cols = ["no_of_employees", "prevailing_wage", "company_age"]
            cat_cols = self.schema_config["categorical_columns"]
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numeric_transformer, num_cols),
                    ("cat", oh_transformer, cat_cols)
                ]
            )
            return preprocessor
        except Exception as e:
            raise USVisaException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Starting Data Transformation")
            
            if not self.data_validation_artifact.validation_status:
                raise Exception("Data Validation failed. Cannot start Data Transformation.")
                
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            
            target_column = self.schema_config["target_column"]
            
            # Create feature engineering columns
            current_year = datetime.now().year
            train_df["company_age"] = current_year - train_df["yr_of_estab"]
            test_df["company_age"] = current_year - test_df["yr_of_estab"]
            
            # Define drop columns
            drop_cols = self.schema_config["drop_columns"] + ["yr_of_estab"]
            
            # Separate features and target
            input_feature_train_df = train_df.drop(columns=drop_cols + [target_column], errors="ignore")
            target_feature_train_df = train_df[target_column]
            
            input_feature_test_df = test_df.drop(columns=drop_cols + [target_column], errors="ignore")
            target_feature_test_df = test_df[target_column]
            
            # Map target variable (Certified -> 1, Denied -> 0)
            target_feature_train_df = target_feature_train_df.map({"Certified": 1, "Denied": 0})
            target_feature_test_df = target_feature_test_df.map({"Certified": 1, "Denied": 0})
            
            preprocessor = self.get_data_transformer_object()
            
            logging.info("Applying ColumnTransformer on train and test features")
            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)
            
            logging.info("Applying SMOTE to handle target class imbalance on training dataset")
            smote = SMOTE(random_state=42)
            input_feature_train_final, target_feature_train_final = smote.fit_resample(
                input_feature_train_arr, target_feature_train_df
            )
            
            # Combine features and targets
            train_arr = np.c_[input_feature_train_final, np.array(target_feature_train_final)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            
            # Save transformed datasets
            save_numpy_array_data(
                file_path=self.data_transformation_config.transformed_train_file_path,
                array=train_arr
            )
            save_numpy_array_data(
                file_path=self.data_transformation_config.transformed_test_file_path,
                array=test_arr
            )
            
            # Save preprocessor object
            save_object(
                file_path=self.data_transformation_config.transformed_object_file_path,
                obj=preprocessor
            )
            
            data_transformation_artifact = DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path
            )
            logging.info(f"Data Transformation completed. Artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise USVisaException(e, sys)
