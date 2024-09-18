import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_transformer_object(self):
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            numerical_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            categorical_pipeline = Pipeline(
                steps = [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            preprocessor = ColumnTransformer(
                transformers = [
                    ("num_pipeline", numerical_pipeline, numerical_columns),
                    ("cat_pipeline", categorical_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        except Exception as e:
            logging.error(f"Error in preprocessing: {str(e)}")
            raise CustomException(f"Error in preprocessing: {str(e)}", sys.exc_info())

    def initiate_data_transformer(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read Train and Test Data")

            logging.info("Obtaining Preprocessor Object")
            preprocessor = self.get_transformer_object()

            target_column = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            input_feature_train_df = train_df.drop(target_column, axis=1)
            target_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(target_column, axis=1)
            target_test_df = test_df[target_column]

            logging.info("Fitting the Preprocessor Object")

            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_test_df)]

            logging.info("Data Transformation Completed")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor
            )

            return (train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path)
        except Exception as e:
            logging.error(f"Error in data transformation: {str(e)}")
            raise CustomException(f"Error in data transformation: {str(e)}", sys.exc_info())            