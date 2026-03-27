import os
import pickle
import sys
import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)