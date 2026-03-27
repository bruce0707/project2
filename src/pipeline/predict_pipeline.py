import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import pickle
from src.logger import logging

class PredictPipeline:
    def __init__(self):
        try:
            self.model = pickle.load(open("artifacts/model.pkl", "rb"))
            self.preprocessor = pickle.load(open("artifacts/preprocessor.pkl", "rb"))
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, features: pd.DataFrame):
        try:
            logging.info("Starting prediction pipeline")

            
            data_scaled = self.preprocessor.transform(features)

            # Prediction
            prediction = self.model.predict(data_scaled)[0]

            
            probability = self.model.predict_proba(data_scaled)[0][1]

            
            risk_level = self.get_risk_level(probability)

           
            suggestions = self.get_suggestions(features.iloc[0])

            return {
                "prediction": int(prediction),
                "probability": round(probability, 2),
                "risk_level": risk_level,
                "suggestions": suggestions
            }

        except Exception as e:
            raise CustomException(e, sys)

    
    def get_risk_level(self, prob):
        if prob < 0.3:
            return "Low 🟢"
        elif prob < 0.7:
            return "Medium 🟡"
        else:
            return "High 🔴"

    
    def get_suggestions(self, data):
        tips = []

        try:
            
            if "chol" in data and float(data["chol"]) > 240:
                tips.append("Reduce oily food and cholesterol intake")

            
            if "trestbps" in data and float(data["trestbps"]) > 140:
                tips.append("Control blood pressure (reduce salt)")

            
            if "thalch" in data and float(data["thalch"]) < 100:
                tips.append("Increase physical activity")

            
            if "oldpeak" in data and float(data["oldpeak"]) > 2:
                tips.append("Consult doctor for heart stress issues")

            
            if "cp" in data and float(data["cp"]) > 2:
                tips.append("Monitor chest pain and consult doctor")

            if len(tips) == 0:
                tips.append("All parameters look normal. Maintain healthy lifestyle 👍")

            return tips

        except Exception as e:
            return ["Unable to generate suggestions"]

        

class CustomData:
    def __init__(self,
                 age: float,
                 sex: int,
                 cp: int,
                 trestbps: float,
                 chol: float,
                 fbs: int,
                 restecg: int,
                 thalch: float,
                 exang: int,
                 oldpeak: float,
                 slope: int,
                 ca: int,
                 thal: int):
        
        self.age = age
        self.sex = sex
        self.cp = cp
        self.trestbps = trestbps
        self.chol = chol
        self.fbs = fbs
        self.restecg = restecg
        self.thalch = thalch
        self.exang = exang
        self.oldpeak = oldpeak
        self.slope = slope
        self.ca = ca
        self.thal = thal

    def get_data_as_dataframe(self):
        try:
            data_dict = {
                "age": [self.age],
                "sex": [self.sex],
                "cp": [self.cp],
                "trestbps": [self.trestbps],
                "chol": [self.chol],
                "fbs": [self.fbs],
                "restecg": [self.restecg],
                "thalch": [self.thalch],
                "exang": [self.exang],
                "oldpeak": [self.oldpeak],
                "slope": [self.slope],
                "ca": [self.ca],
                "thal": [self.thal]
            }

            return pd.DataFrame(data_dict)

        except Exception as e:
            raise CustomException(e, sys)