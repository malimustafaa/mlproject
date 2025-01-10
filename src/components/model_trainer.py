import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
        RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("splitting train and test data")
            X_train,Y_train,X_test,Y_test = (

                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regressor": LinearRegression(),
                "K-Neighbors Regresssor": KNeighborsRegressor(),
                "Adaboost Regressor":AdaBoostRegressor()
            }
            model_report:dict= evaluate_models(X_train,Y_train, X_test, Y_test,models)

            #to get best model score from dict
            best_model_score = max(sorted(model_report.values()))
            # to get bestmodel name from dict
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            if best_model_score <0.6:
                raise CustomException("no best model found")
            logging.info("best found model on both trainng and test data")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj = best_model

            )
            predicted = best_model.predict(X_test)
            r2score = r2_score(Y_test,predicted)
            return r2score
        

        except Exception as e:
            raise CustomException(e,sys)
    


