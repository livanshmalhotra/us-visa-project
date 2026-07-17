import os
import sys
import shutil
from us_visa.entity.config_entity import ModelPusherConfig
from us_visa.entity.artifact_entity import ModelPusherArtifact, ModelEvaluationArtifact
from us_visa.exception import USVisaException
from us_visa.logger import logging

class ModelPusher:
    def __init__(self, model_pusher_config: ModelPusherConfig, model_evaluation_artifact: ModelEvaluationArtifact):
        try:
            self.model_pusher_config = model_pusher_config
            self.model_evaluation_artifact = model_evaluation_artifact
        except Exception as e:
            raise USVisaException(e, sys)

    def initiate_model_pusher(self) -> ModelPusherArtifact:
        try:
            logging.info("Starting Model Pusher component")
            
            if self.model_evaluation_artifact.is_model_accepted:
                trained_model_path = self.model_evaluation_artifact.trained_model_path
                saved_model_path = self.model_pusher_config.saved_model_file_path
                
                os.makedirs(os.path.dirname(saved_model_path), exist_ok=True)
                
                logging.info(f"Copying accepted model from {trained_model_path} to production folder: {saved_model_path}")
                shutil.copy(src=trained_model_path, dst=saved_model_path)
                
                model_pusher_artifact = ModelPusherArtifact(saved_model_path=saved_model_path)
                logging.info(f"Model successfully pushed. Artifact: {model_pusher_artifact}")
                return model_pusher_artifact
            else:
                logging.info("Trained model was rejected during evaluation. Model is not pushed to production.")
                model_pusher_artifact = ModelPusherArtifact(saved_model_path="")
                return model_pusher_artifact
        except Exception as e:
            raise USVisaException(e, sys)
