# import os
# import sys
# import mlflow
# import mlflow.sklearn
# import numpy as np
# from urllib.parse import urlparse
# from dataclasses import dataclass
# from src.mlproject.utils import save_object
# from src.mlproject.utils import evaluate_models 
# from src.mlproject.exception import CustomException
# from src.mlproject.logger import logging
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import train_test_split
# from xgboost import XGBRegressor
# from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# from sklearn.linear_model import LinearRegression
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import (
#     RandomForestRegressor,
#     GradientBoostingRegressor,
#     AdaBoostRegressor,
#     )


# @dataclass
# class ModelTrainerConfig:
#     trained_model_file_path = os.path.join("artifacts", "model.pkl")

# class ModelTrainer:
#     def __init__(self):
#         self.model_trainer_config = ModelTrainerConfig()

#     def eval_metrics(self, actual, pred):
#         mae = mean_absolute_error(actual, pred)
#         rmse = np.sqrt(mean_squared_error(actual, pred))
#         r2 = r2_score(actual, pred)
#         return mae, rmse, r2
    
#     def initiate_model_trainer(self, train_array, test_array):
#         try:
#             logging.info("Splitting training and test input data")
#             X_train, y_train, X_test, y_test = (
#                 train_array[:,:-1],
#                 train_array[:,-1],
#                 test_array[:,:-1],
#                 test_array[:,-1]
#             )

#             models = {  
#                 "Random Forest": RandomForestRegressor(),
#                 "Decision Tree": DecisionTreeRegressor(),
#                 "Gradient Boosting": GradientBoostingRegressor(),
#                 "Linear Regression": LinearRegression(),
#                 "XGBRegressor": XGBRegressor(),
#                 "KNN Regressor": KNeighborsRegressor(),
#                 "AdaBoost Regressor": AdaBoostRegressor()
#             }

#             params = {
#                 "Random Forest": {
#                     'n_estimators': [8, 16, 32, 64, 128, 256],
#                     'max_depth': [4, 8, 12, 16, 20, 24]
#                 },
#                 "Decision Tree": {
#                     'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'], 
#                     'splitter': ['best', 'random']
#                 },  
#                 "Gradient Boosting": {
#                     'n_estimators': [8, 16, 32, 64, 128, 256],
#                     'max_depth': [4, 8, 12, 16, 20, 24]
#                 },  
#                 "Linear Regression": {},
#                 "XGBRegressor": {
#                     'learning_rate':[.1,.01,.05,.001],
#                     'n_estimators': [8,16,32,64,128,256]
#                 },  
#                 "KNN Regressor": {
#                     'n_neighbors': [5, 7, 9, 11, 13, 15],
#                     'weights': ['uniform', 'distance']
#                 },  
#                 "AdaBoost Regressor": {
#                     'n_estimators': [8, 16, 32, 64, 128, 256],
#                     'learning_rate': [1, 0.5, 0.1, 0.01, 0.001]
#                 }
#             }

#             model_report: dict = evaluate_models(X_train, y_train, X_test, y_test, models, params)  

#             best_model_score = max(sorted(model_report.values()))

#             best_model_name = list(model_report.keys())[
#                 list(model_report.values()).index(best_model_score)
#             ]

#             best_model = models[best_model_name]  

#             print("This is the best model...")
#             print(best_model_name)

#             model_names =  list(params.keys())

#             actual_model = ""

#             for model in model_names:
#                 if best_model_name == model:
#                     actual_model = actual_model + model

#             best_params =  params[actual_model]

#             # MLflow Tracking Setup
#             mlflow.set_registry_uri("https://dagshub.com/muhammadadilnaeem/Student-Performance-Indicater-End-To-End-Data-Science-Project.mlflow")
#             tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

#             # MLflow Pipeline
#             logging.info("Now we are going to use MLflow to log our best model")

#             with mlflow.start_run():
#                 predicted_qualities = best_model.predict(X_test)
#                 (rmse, mae, r2) = self.eval_metrics(y_test, predicted_qualities)

#                 mlflow.log_params(best_params)
#                 mlflow.log_metric("rmse", rmse)
#                 mlflow.log_metric("r2", r2)
#                 mlflow.log_metric("mae", mae)

#                 if tracking_url_type_store != "file":
#                     mlflow.sklearn.log_model(best_model, "model", registered_model_name=actual_model)
#                 else:
#                     mlflow.sklearn.log_model(best_model, "model")
            
#             if best_model_score < 0.6:
#                 raise CustomException("No best model found")

#             logging.info(f"Best found model on both training and testing dataset")  

#             save_object(
#                 file_path=self.model_trainer_config.trained_model_file_path,
#                 obj=best_model
#             )

#             predicted = best_model.predict(X_test)
#             r2_square = r2_score(y_test, predicted)

#             return r2_square
        
#         except Exception as e:
#             raise CustomException(e, sys)

import os
import sys
import mlflow
import mlflow.sklearn
import numpy as np
from urllib.parse import urlparse
from dataclasses import dataclass
from src.mlproject.utils import save_object
from src.mlproject.utils import evaluate_models
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
import dagshub
from mlflow.models import infer_signature

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def eval_metrics(self, actual, pred):
        mae = mean_absolute_error(actual, pred)
        rmse = np.sqrt(mean_squared_error(actual, pred))
        r2 = r2_score(actual, pred)
        return mae, rmse, r2

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "KNN Regressor": KNeighborsRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            params = {
                "Random Forest": {
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'max_depth': [4, 8, 12, 16, 20, 24]
                },
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter': ['best', 'random']
                },
                "Gradient Boosting": {
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'max_depth': [4, 8, 12, 16, 20, 24]
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    'learning_rate': [.1, .01, .05, .001],
                    'n_estimators': [8, 16, 32, 64, 128, 256]
                },
                "KNN Regressor": {
                    'n_neighbors': [5, 7, 9, 11, 13, 15],
                    'weights': ['uniform', 'distance']
                },
                "AdaBoost Regressor": {
                    'n_estimators': [8, 16, 32, 64, 128, 256],
                    'learning_rate': [1, 0.5, 0.1, 0.01, 0.001]
                }
            }

            model_report: dict = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            print("This is the best model...")
            print(best_model_name)

            model_names = list(params.keys())
            actual_model = ""

            for model in model_names:
                if best_model_name == model:
                    actual_model = actual_model + model

            best_params = params[actual_model]

            # Initialize DagsHub with MLflow integration
            dagshub.init(repo_owner='muhammadadilnaeem', repo_name='Student-Performance-Indicater-End-To-End-Data-Science-Project', mlflow=True)

            # Set MLflow tracking URI
            mlflow.set_tracking_uri("https://dagshub.com/muhammadadilnaeem/Student-Performance-Indicater-End-To-End-Data-Science-Project.mlflow")
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            # MLflow Pipeline
            logging.info("Now we are going to use MLflow to log our best model")

            with mlflow.start_run() as run:
                print("Run ID:", run.info.run_id)

                predicted_qualities = best_model.predict(X_test)
                (rmse, mae, r2) = self.eval_metrics(y_test, predicted_qualities)

                mlflow.log_params(best_params)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("r2", r2)
                mlflow.log_metric("mae", mae)

                # Add model signature
                signature = infer_signature(X_test, predicted_qualities)
                mlflow.sklearn.log_model(best_model, "model", signature=signature)

                if tracking_url_type_store != "file":
                    mlflow.sklearn.log_model(best_model, "model", registered_model_name=actual_model)
                else:
                    mlflow.sklearn.log_model(best_model, "model")

            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
