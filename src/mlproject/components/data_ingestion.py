import os
import sys
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from src.mlproject.utils import read_sql_data
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import pandas as pd

# Define data ingestion configuration class for reading the data from MySQL database
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "raw.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    # Let's read the data from MySQL database
    def initiate_data_ingestion(self):
        try:
            sql_df = read_sql_data()
            logging.info("Reading data from MySQL database has been completed.")
            
            # Create directory if it does not exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save data from SQL Database and save it to csv as raw.csv
            sql_df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            # Split the saved data to train.csv and test.csv
            train_set, test_set = train_test_split(sql_df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            # Ingestion of data is completed
            logging.info("Ingestion of data is completed.")

            # return the path of train.csv and test.csv
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        # Let's handle the exception
        except Exception as e:
            raise CustomException(e, sys)
