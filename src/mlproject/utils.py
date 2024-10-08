import os
import sys
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from dataclasses import dataclass
import pandas as pd
import pymysql

import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score


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
    
# Function to evaluate models using GridSearchCV

def evaluate_models(X_train, y_train,X_test,y_test,models,param):
    try:
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            para=param[list(models.keys())[i]]

            gs = GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train,y_train)

            #model.fit(X_train, y_train)  # Train model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)

            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)