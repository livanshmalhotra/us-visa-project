import os
import sys
import pymongo
import certifi
from us_visa.constants import DATABASE_NAME, MONGODB_URL_KEY
from us_visa.exception import USVisaException
from us_visa.logger import logging

# Load .env file manually if python-dotenv is not installed
if os.path.exists(".env"):
    with open(".env") as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                key, val = line.strip().split("=", 1)
                os.environ[key.strip()] = val.strip()

ca = certifi.where()

class MongoDBClient:
    client = None
    
    def __init__(self, database_name=DATABASE_NAME) -> None:
        try:
            if MongoDBClient.client is None:
                mongo_db_url = os.getenv(MONGODB_URL_KEY) or os.getenv("MONGODB_URI")
                if mongo_db_url is None:
                    raise Exception(f"Environment variable '{MONGODB_URL_KEY}' or 'MONGODB_URI' is not set.")
                logging.info(f"Connecting to MongoDB with connection string length: {len(mongo_db_url)}")
                MongoDBClient.client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
            
            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name
            logging.info(f"Successfully connected to MongoDB database: {self.database_name}")
        except Exception as e:
            raise USVisaException(e, sys)
