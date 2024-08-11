import sys
from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from src.mlproject.components.data_ingestion import DataIngestion

if __name__ == "__main__":
    logging.info("Logging has been set up.")

    try:
        data_ingestion = DataIngestion()
        data_ingestion.initiate_data_ingestion()

        logging.info("Data ingestion has been completed.")
    except Exception as e:
        logging.info("An error occurred during data ingestion.")
        raise CustomException(e, sys)
