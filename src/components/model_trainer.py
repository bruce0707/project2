import os
import sys
import numpy as np
import pickle
from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from src.exception import CustomException
from src.logger import logging

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def evaluate_models(self, X_train, y_train, X_test, y_test, models, params):
        try:
            report = {}

            for name, model in models.items():
                logging.info(f"Training {name}")

                gs = GridSearchCV(model, params[name], cv=3, n_jobs=-1, verbose=0)
                gs.fit(X_train, y_train)

                best_model = gs.best_estimator_

                y_pred = best_model.predict(X_test)
                y_prob = best_model.predict_proba(X_test)[:, 1]

                acc = accuracy_score(y_test, y_pred)
                roc = roc_auc_score(y_test, y_prob)
                cm = confusion_matrix(y_test, y_pred)

                logging.info(f"{name} Accuracy: {acc}")
                logging.info(f"{name} ROC AUC: {roc}")
                logging.info(f"{name} Confusion Matrix:\n{cm}")

                report[name] = {
                    "model": best_model,
                    "accuracy": acc,
                    "roc_auc": roc
                }

            return report

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Splitting data")

            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            models = {
                        "Random Forest": RandomForestClassifier(),
                        "Decision Tree": DecisionTreeClassifier(),
                        "KNN": KNeighborsClassifier(),
                        "Gradient Boosting": GradientBoostingClassifier(),
                        "XGBoost": XGBClassifier(eval_metric='logloss'),
                        "CatBoost": CatBoostClassifier(verbose=0)
                        }

            params = {
                "Random Forest": {
                    "n_estimators": [50, 100],
                    "max_depth": [None, 10]
                },
                "Decision Tree": {
                    "max_depth": [5, 10, None]
                },
                "KNN": {
                    "n_neighbors": [3, 5, 7]
                },
                "Gradient Boosting": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1]
                },
                "XGBoost": {
                    "n_estimators": [50, 100],
                    "learning_rate": [0.01, 0.1]
                },
                "CatBoost": {
                    "depth": [4, 6],
                    "learning_rate": [0.01, 0.1]
                }
            }

            model_report = self.evaluate_models(
                X_train, y_train, X_test, y_test, models, params
            )

            # 🎯 Select best model based on ROC AUC
            best_model_name = max(model_report, key=lambda x: model_report[x]["roc_auc"])
            best_model = model_report[best_model_name]["model"]
            best_score = model_report[best_model_name]["roc_auc"]

            logging.info(f"Best Model: {best_model_name} with ROC AUC: {best_score}")

            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)

            with open(self.model_trainer_config.trained_model_file_path, "wb") as f:
                pickle.dump(best_model, f)

            return best_model_name, best_score

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    from src.components.data_transformation import DataTransformation

    dt = DataTransformation()

    train_arr, test_arr, _ = dt.initiate_data_transformation(
        "artifacts/train_data.csv",
        "artifacts/test_data.csv"
    )

    mt = ModelTrainer()

    model_name, score = mt.initiate_model_trainer(train_arr, test_arr)

    print(f"Best Model: {model_name}")
    print(f"Best ROC AUC: {score}")