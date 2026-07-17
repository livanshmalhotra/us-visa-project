import os
import sys
import numpy as np
import importlib
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from us_visa.entity.config_entity import ModelTrainerConfig
from us_visa.entity.artifact_entity import ModelTrainerArtifact, ClassificationMetricArtifact
from us_visa.exception import USVisaException
from us_visa.logger import logging
from us_visa.utils.main_utils import read_yaml_file, save_object, load_object, load_numpy_array_data

class USVisaModel:
    """
    Unified model wrapper that packs the preprocessing pipeline and the trained ML model together.
    Prevents feature mismatch during training and inference.
    """
    def __init__(self, preprocessing_object, trained_model_object):
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, dataframe):
        try:
            # Transform features using the training preprocessor pipeline
            transformed_feature = self.preprocessing_object.transform(dataframe)
            return self.trained_model_object.predict(transformed_feature)
        except Exception as e:
            raise e

    def __repr__(self):
        return f"USVisaModel(preprocessing_object={type(self.preprocessing_object).__name__}, trained_model_object={type(self.trained_model_object).__name__})"

class ModelTrainer:
    def __init__(self, data_transformation_artifact, model_trainer_config: ModelTrainerConfig):
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config
            self.model_config = read_yaml_file("config/model.yaml")
        except Exception as e:
            raise USVisaException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Starting Model Training component")
            
            # Load transformed training and testing data
            train_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_train_file_path)
            test_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_test_file_path)
            
            # Separate features and target label
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]
            
            model_selection_config = self.model_config.get("model_selection", {})
            
            best_model_obj = None
            best_model_name = None
            best_f1_score = 0.0
            
            # Train and tune multiple classifiers
            for model_key, model_info in model_selection_config.items():
                class_name = model_info["class"]
                module_name = model_info["module"]
                param_grid = model_info["params"]
                
                logging.info(f"Training model: {class_name} from module: {module_name}")
                
                # Import module and class dynamically
                module = importlib.import_module(module_name)
                model_class = getattr(module, class_name)
                
                # Instantiate model
                model = model_class()
                
                # Run Grid Search Cross Validation
                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=param_grid,
                    cv=3,
                    scoring='f1',
                    verbose=1,
                    n_jobs=-1
                )
                grid_search.fit(X_train, y_train)
                
                best_estimator = grid_search.best_estimator_
                
                # Evaluate F1 score on test dataset
                y_pred = best_estimator.predict(X_test)
                score = f1_score(y_test, y_pred)
                
                logging.info(f"Model: {class_name}, Best parameters: {grid_search.best_params_}, Test F1: {score:.4f}")
                
                if score > best_f1_score:
                    best_f1_score = score
                    best_model_obj = best_estimator
                    best_model_name = class_name
            
            if best_model_obj is None:
                raise Exception("No model was successfully trained or selected.")
                
            logging.info(f"Best model selected: {best_model_name} with F1 score: {best_f1_score:.4f}")
            
            # Check overfitting & underfitting constraints
            y_train_pred = best_model_obj.predict(X_train)
            train_f1 = f1_score(y_train, y_train_pred)
            
            f1_diff = abs(train_f1 - best_f1_score)
            logging.info(f"Train F1: {train_f1:.4f}, Test F1: {best_f1_score:.4f}, Score Difference: {f1_diff:.4f}")
            
            if best_f1_score < self.model_trainer_config.expected_accuracy:
                raise Exception(f"Best model F1 score {best_f1_score:.4f} is below minimum threshold: {self.model_trainer_config.expected_accuracy}")
                
            if f1_diff > self.model_trainer_config.overfitting_limit:
                logging.warning(f"Overfitting detected! F1 score difference is {f1_diff:.4f} which is higher than overfitting limit {self.model_trainer_config.overfitting_limit}")
            
            # Load preprocessing ColumnTransformer object
            preprocessor = load_object(self.data_transformation_artifact.transformed_object_file_path)
            
            # Wrap preprocessor and best model together
            logging.info("Wrapping preprocessor and model inside USVisaModel")
            us_visa_model = USVisaModel(preprocessing_object=preprocessor, trained_model_object=best_model_obj)
            
            # Save final wrapped model to artifact dir
            trained_model_file_path = self.model_trainer_config.trained_model_file_path
            os.makedirs(os.path.dirname(trained_model_file_path), exist_ok=True)
            save_object(file_path=trained_model_file_path, obj=us_visa_model)
            
            # Calculate classification metrics
            y_test_pred = best_model_obj.predict(X_test)
            test_metric = ClassificationMetricArtifact(
                f1_score=f1_score(y_test, y_test_pred),
                precision_score=precision_score(y_test, y_test_pred),
                recall_score=recall_score(y_test, y_test_pred)
            )
            
            train_metric = ClassificationMetricArtifact(
                f1_score=f1_score(y_train, y_train_pred),
                precision_score=precision_score(y_train, y_train_pred),
                recall_score=recall_score(y_train, y_train_pred)
            )
            
            trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=trained_model_file_path,
                train_metric_artifact=train_metric,
                test_metric_artifact=test_metric
            )
            logging.info(f"Model Trainer completed. Artifact: {trainer_artifact}")
            return trainer_artifact
        except Exception as e:
            raise USVisaException(e, sys)
