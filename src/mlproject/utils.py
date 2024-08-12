import os
import sys
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from dataclasses import dataclass
import pandas as pd
import pymysql

import pickle


# Setting up environment variables
from dotenv import load_dotenv
load_dotenv()

host = os.getenv("host")
user = os.getenv("user")
password = os.getenv("password")
db = os.getenv("db")

# Read SQL DataFrame
def read_sql_data():
    logging.info("Reading data from MySQL database.")
    try:
        conn = pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db  # Ensure this is the correct database name
        )
        logging.info("Connection to MySQL database has been established.")
        sql_df = pd.read_sql_query("SELECT * FROM students", conn)
        print(sql_df.head())
        return sql_df

    except Exception as ex:
        raise CustomException(ex)
    
# Function to save Pickle file
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path,"wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)