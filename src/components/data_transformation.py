import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from src.components.data_ingestion import DataIngestion
from src.exception import CustomException
from src.logger import logging

from sklearn.model_selection import train_test_split
import pickle


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            logging.info("Creating preprocessing pipelines")

            numerical_columns = [
                                "age", "trestbps", "chol", "thalch", "oldpeak"
                                     ]

            categorical_columns = [
                                    "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"
        ]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore"))
                ]
            )

            logging.info("Pipelines created successfully")

            
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Starting data transformation")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Data loaded successfully")

            train_df.drop(columns=["id"], inplace=True, errors="ignore")
            test_df.drop(columns=["id"], inplace=True, errors="ignore")

            train_df.replace("?", np.nan, inplace=True)
            test_df.replace("?", np.nan, inplace=True)
            train_df = train_df.apply(pd.to_numeric, errors='ignore')
            test_df = test_df.apply(pd.to_numeric, errors='ignore')

            train_df["num"] = train_df["num"].apply(lambda x: 1 if x > 0 else 0)
            test_df["num"] = test_df["num"].apply(lambda x: 1 if x > 0 else 0)
            target_column_name = "num"

            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]

            preprocessor = self.get_data_transformer_object()

            logging.info("Applying preprocessing on train and test data")

            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info("Saving preprocessing object")

            with open(self.data_transformation_config.preprocessor_obj_file_path, "wb") as f:
                pickle.dump(preprocessor, f)

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
        

if __name__ == "__main__":

    obj = DataTransformation()

    train_arr, test_arr, path = obj.initiate_data_transformation(
        "artifacts/train_data.csv",
        "artifacts/test_data.csv"
    )

    print("Preprocessor saved at:", path)
    logging.info("Data transformation completed successfully")