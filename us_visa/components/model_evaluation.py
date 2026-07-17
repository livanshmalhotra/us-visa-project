import os
import sys
import pandas as pd
from datetime import datetime
from sklearn.metrics import f1_score
from us_visa.entity.config_entity import ModelEvaluationConfig
from us_visa.entity.artifact_entity import ModelEvaluationArtifact, ModelTrainerArtifact, DataIngestionArtifact
from us_visa.exception import USVisaException
from us_visa.logger import logging
from us_visa.utils.main_utils import read_yaml_file, write_yaml_file, load_object
from us_visa.constants import SCHEMA_FILE_PATH, MODEL_FILE_NAME

class ModelEvaluation:
    def __init__(self, model_evaluation_config: ModelEvaluationConfig,
                 data_ingestion_artifact: DataIngestionArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_evaluation_config = model_evaluation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise USVisaException(e, sys)

    def get_best_model_path(self) -> str:
        """
        Method Name : get_best_model_path
        Description : Locates the currently active model pushed to production folder (if any)
        """
        try:
            prod_model_path = os.path.join("models", MODEL_FILE_NAME)
            if os.path.exists(prod_model_path):
                return prod_model_path
            return None
        except Exception as e:
            raise USVisaException(e, sys)

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            logging.info("Starting Model Evaluation component")
            
            # Load raw testing data
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            target_column = self.schema_config["target_column"]
            
            # Preprocess features: calculate company age and drop drop_columns
            current_year = datetime.now().year
            test_df["company_age"] = current_year - test_df["yr_of_estab"]
            drop_cols = self.schema_config["drop_columns"] + ["yr_of_estab"]
            
            X_test = test_df.drop(columns=drop_cols + [target_column], errors="ignore")
            y_test = test_df[target_column].map({"Certified": 1, "Denied": 0})
            
            # Load newly trained USVisaModel
            trained_model_path = self.model_trainer_artifact.trained_model_file_path
            trained_model = load_object(trained_model_path)
            
            # Predict using newly trained model
            trained_model_preds = trained_model.predict(X_test)
            trained_model_f1 = f1_score(y_test, trained_model_preds)
            
            prod_model_path = self.get_best_model_path()
            
            # If no active model exists in production, accept the newly trained model automatically
            if prod_model_path is None:
                logging.info("No production model found. Automatically accepting the newly trained model.")
                is_model_accepted = True
                improved_score = trained_model_f1
                prod_model_f1 = 0.0
            else:
                logging.info(f"Production model found at: {prod_model_path}. Comparing scores...")
                prod_model = load_object(prod_model_path)
                prod_model_preds = prod_model.predict(X_test)
                prod_model_f1 = f1_score(y_test, prod_model_preds)
                
                improved_score = trained_model_f1 - prod_model_f1
                logging.info(f"Trained model F1: {trained_model_f1:.4f}, Production model F1: {prod_model_f1:.4f}, Improvement: {improved_score:.4f}")
                
                if improved_score >= self.model_evaluation_config.changed_threshold_score:
                    is_model_accepted = True
                    logging.info(f"Newly trained model accepted. Improvement score is above threshold: {self.model_evaluation_config.changed_threshold_score}")
                else:
                    is_model_accepted = False
                    logging.info(f"Newly trained model rejected. Improvement score is below threshold: {self.model_evaluation_config.changed_threshold_score}")
            
            # Save evaluation report
            eval_report = {
                "trained_model_path": trained_model_path,
                "production_model_path": prod_model_path if prod_model_path else "None",
                "trained_model_f1": float(trained_model_f1),
                "production_model_f1": float(prod_model_f1),
                "improvement": float(improved_score),
                "is_model_accepted": is_model_accepted
            }
            write_yaml_file(file_path=self.model_evaluation_config.report_file_path, content=eval_report, replace=True)
            
            model_evaluation_artifact = ModelEvaluationArtifact(
                is_model_accepted=is_model_accepted,
                changed_threshold_score=self.model_evaluation_config.changed_threshold_score,
                trained_model_path=trained_model_path,
                best_model_path=prod_model_path if prod_model_path else ""
            )
            logging.info(f"Model Evaluation completed. Artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact
        except Exception as e:
            raise USVisaException(e, sys)
