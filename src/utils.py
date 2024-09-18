import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
import dill
from src.logger import logging
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file:
            dill.dump(obj, file)

    except Exception as e:
        logging.error(f"Error in saving object: {str(e)}")
        raise CustomException(f"Error in saving object: {str(e)}", sys.exc_info())

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    try:
        model_report = {}
        for i in range(len(list(models))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            para = params[list(models.keys())[i]]

            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)

            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            model_report[list(models.keys())[i]] = test_model_score

            # logging.info(f"{model_name} trained successfully")
            # logging.info(f"Train Score: {train_model_score}")
            # logging.info(f"Test Score: {test_model_score}")

        return model_report
    except Exception as e:
        logging.error(f"Error in evaluating models: {str(e)}")
        raise CustomException(f"Error in evaluating models: {str(e)}", sys.exc_info())


def load_object(file_path):
    try:
        with open(file_path, "rb") as file:
            obj = dill.load(file)
        return obj
    except Exception as e:
        logging.error(f"Error in loading object: {str(e)}")
        raise CustomException(f"Error in loading object: {str(e)}", sys.exc_info())