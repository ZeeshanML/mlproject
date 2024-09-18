import os, sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing data")

            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "XGBoost": XGBRegressor(),
                "KNN": KNeighborsRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False)
            }

            model_report:dict = evaluate_models(X_train = X_train,
                                               y_train = y_train,
                                               X_test = X_test,
                                               y_test = y_test,
                                               models = models)

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("Best Model Score is less than 0.6", sys.exc_info())

            logging.info(f"Best Model: {best_model_name}")

            save_object(self.model_trainer_config.trained_model_file_path, best_model)

            logging.info("Model trained and saved successfully")

            pred = best_model.predict(X_test)
            r2 = r2_score(y_test, pred)

            return r2

        except Exception as e:
            logging.error(f"Error in training model: {str(e)}")
            raise CustomException(f"Error in training model: {str(e)}", sys.exc_info())