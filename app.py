import sys
from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from src.mlproject.components.data_ingestion import DataIngestion
from src.mlproject.components.data_transformation import DataTransformation

if __name__ == "__main__":
    logging.info("Logging has been set up.")

    try:
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        logging.info("Data ingestion has been completed.")

        data_transformation = DataTransformation()
        data_transformation.initiate_data_transformation(train_data_path, test_data_path)
        logging.info("Data transformation has been completed.")

    except Exception as e:
        logging.info("An error occurred during data ingestion.")
        raise CustomException(e, sys)
