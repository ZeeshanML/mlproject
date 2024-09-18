import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
import dill
from src.logger import logging


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file:
            dill.dump(obj, file)
            
    except Exception as e:
        logging.error(f"Error in saving object: {str(e)}")
        raise CustomException(f"Error in saving object: {str(e)}", sys.exc_info())
