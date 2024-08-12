import os
import sys
from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from src.mlproject.components.data_ingestion import DataIngestion
from src.mlproject.components.data_transformation import DataTransformation
from src.mlproject.components.model_trainer import ModelTrainer, ModelTrainerConfig


if __name__ == "__main__":
    logging.info("Logging has been set up.")

    try:
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        logging.info("Data ingestion has been completed.")

        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data_path, test_data_path)
        logging.info("Data transformation has been completed.")

        model_trainer = ModelTrainer()
        print(model_trainer.initiate_model_trainer(train_arr, test_arr))
        logging.info("Model training has been completed.") 

    except Exception as e:
        logging.info("An error occurred during data ingestion.")
        raise CustomException(e, sys)
