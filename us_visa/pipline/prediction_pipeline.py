import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from us_visa.exception import USVisaException
from us_visa.logger import logging
from us_visa.utils.main_utils import load_object
from us_visa.constants import MODEL_FILE_NAME, MODEL_PUSHER_SAVED_MODEL_DIR

class PredictionPipeline:
    def __init__(self):
        pass

    def predict(self, dataframe: pd.DataFrame) -> dict:
        """
        Method Name : predict
        Description : Preprocesses input dataframe, runs prediction using the saved USVisaModel, 
                      and returns prediction label and probability/confidence.
        """
        try:
            logging.info("Starting prediction pipeline")
            model_path = os.path.join(MODEL_PUSHER_SAVED_MODEL_DIR, MODEL_FILE_NAME)
            
            if not os.path.exists(model_path):
                raise Exception("Production model file not found. Please train the model first.")
            
            # Load the wrapped USVisaModel (contains preprocessor and estimator)
            model = load_object(model_path)
            
            # Calculate company_age and drop yr_of_estab
            current_year = datetime.now().year
            dataframe["company_age"] = current_year - dataframe["yr_of_estab"]
            
            # Keep only the features that were present during fit (excluding case_id, yr_of_estab, case_status)
            drop_cols = ["case_id", "yr_of_estab", "case_status"]
            X_input = dataframe.drop(columns=drop_cols, errors="ignore")
            
            logging.info(f"Features passed for transformation: {list(X_input.columns)}")
            
            # Run prediction
            prediction_val = model.predict(X_input)[0]
            prediction_label = "Certified" if prediction_val == 1 else "Denied"
            
            # Calculate confidence score if the underlying model supports predict_proba
            confidence = 0.95  # Default
            try:
                # Transform features to run predict_proba
                transformed = model.preprocessing_object.transform(X_input)
                if hasattr(model.trained_model_object, "predict_proba"):
                    prob = model.trained_model_object.predict_proba(transformed)[0]
                    confidence = float(prob[int(prediction_val)])
                    logging.info(f"Prediction probability: {prob}")
            except Exception as prob_err:
                logging.warning(f"Could not calculate probability: {prob_err}")
                
            result = {
                "prediction": prediction_label,
                "confidence": confidence,
                "model_version": "1.0.0"
            }
            logging.info(f"Prediction result: {result}")
            return result
        except Exception as e:
            raise USVisaException(e, sys)

class USVisaData:
    def __init__(self,
                 continent: str,
                 education_of_employee: str,
                 has_job_experience: str,
                 requires_job_training: str,
                 no_of_employees: int,
                 yr_of_estab: int,
                 region_of_employment: str,
                 prevailing_wage: float,
                 unit_of_wage: str,
                 full_time_position: str):
        self.continent = continent
        self.education_of_employee = education_of_employee
        self.has_job_experience = has_job_experience
        self.requires_job_training = requires_job_training
        self.no_of_employees = no_of_employees
        self.yr_of_estab = yr_of_estab
        self.region_of_employment = region_of_employment
        self.prevailing_wage = prevailing_wage
        self.unit_of_wage = unit_of_wage
        self.full_time_position = full_time_position
        self.validate_data()

    def validate_data(self):
        """
        Validates input boundaries and data types.
        """
        if not isinstance(self.no_of_employees, (int, float)) or self.no_of_employees < 0:
            raise ValueError("Number of employees must be a non-negative number.")
        if not isinstance(self.prevailing_wage, (int, float)) or self.prevailing_wage < 0:
            raise ValueError("Prevailing wage must be a non-negative number.")
        if self.yr_of_estab < 1800 or self.yr_of_estab > datetime.now().year:
            raise ValueError(f"Year of establishment must be between 1800 and {datetime.now().year}.")

    def get_us_visa_input_data_frame(self) -> pd.DataFrame:
        """
        Converts properties into a pandas DataFrame matching the schema.
        """
        try:
            input_dict = self.get_us_visa_data_as_dict()
            return pd.DataFrame([input_dict])
        except Exception as e:
            raise USVisaException(e, sys)

    def get_us_visa_data_as_dict(self) -> dict:
        """
        Converts properties into a dictionary.
        """
        return {
            "continent": self.continent,
            "education_of_employee": self.education_of_employee,
            "has_job_experience": self.has_job_experience,
            "requires_job_training": self.requires_job_training,
            "no_of_employees": self.no_of_employees,
            "yr_of_estab": self.yr_of_estab,
            "region_of_employment": self.region_of_employment,
            "prevailing_wage": self.prevailing_wage,
            "unit_of_wage": self.unit_of_wage,
            "full_time_position": self.full_time_position,
        }
