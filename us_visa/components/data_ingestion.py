import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from us_visa.configuration.mongo_db_connection import MongoDBClient
from us_visa.entity.config_entity import DataIngestionConfig
from us_visa.entity.artifact_entity import DataIngestionArtifact
from us_visa.exception import USVisaException
from us_visa.logger import logging

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
            self.mongo_client = MongoDBClient()
        except Exception as e:
            raise USVisaException(e, sys)

    def export_data_into_feature_store(self) -> pd.DataFrame:
        """
        Method Name : export_data_into_feature_store
        Description : Exports data from MongoDB Atlas database collection into feature store
        """
        try:
            logging.info("Exporting data from MongoDB to feature store")
            collection = self.mongo_client.database[self.data_ingestion_config.collection_name]
            
            # Seeding database if empty
            if collection.count_documents({}) == 0:
                logging.info("MongoDB collection is empty. Seeding from Rochita's repository URL...")
                dataset_url = "https://raw.githubusercontent.com/rochitasundar/Customer-profiling-using-ML-EasyVisa/master/EasyVisa.csv"
                df_seed = pd.read_csv(dataset_url)
                records = list(df_seed.T.to_dict().values())
                collection.insert_many(records)
                logging.info(f"Seeded {len(records)} documents into MongoDB collection: {self.data_ingestion_config.collection_name}")
            
            # Fetch data from MongoDB
            cursor = collection.find()
            df = pd.DataFrame(list(cursor))
            if "_id" in df.columns:
                df = df.drop(columns=["_id"])
            
            # Save raw dataset to feature store
            raw_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(raw_file_path)
            os.makedirs(dir_path, exist_ok=True)
            df.to_csv(raw_file_path, index=False, header=True)
            logging.info(f"Exported data saved at: {raw_file_path}")
            return df
        except Exception as e:
            raise USVisaException(e, sys)

    def split_data_as_train_test(self, df: pd.DataFrame) -> None:
        """
        Method Name : split_data_as_train_test
        Description : Splits the dataframe into train and test sets
        """
        try:
            logging.info("Splitting data into train and test sets")
            train_set, test_set = train_test_split(
                df, 
                test_size=self.data_ingestion_config.train_test_split_ratio, 
                random_state=42
            )
            
            # Save train data
            train_file_path = self.data_ingestion_config.train_file_path
            os.makedirs(os.path.dirname(train_file_path), exist_ok=True)
            train_set.to_csv(train_file_path, index=False, header=True)
            
            # Save test data
            test_file_path = self.data_ingestion_config.test_file_path
            os.makedirs(os.path.dirname(test_file_path), exist_ok=True)
            test_set.to_csv(test_file_path, index=False, header=True)
            
            logging.info("Train and test data saved successfully.")
        except Exception as e:
            raise USVisaException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Initiating Data Ingestion component")
            df = self.export_data_into_feature_store()
            self.split_data_as_train_test(df)
            
            data_ingestion_artifact = DataIngestionArtifact(
                train_file_path=self.data_ingestion_config.train_file_path,
                test_file_path=self.data_ingestion_config.test_file_path
            )
            logging.info(f"Data Ingestion completed. Artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise USVisaException(e, sys)
