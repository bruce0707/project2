print("🔥 App file is running")
from flask import Flask, request, render_template
import pandas as pd

from src.pipeline.predict_pipeline import PredictPipeline, CustomData

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict_datapoint():
    try:
        data = CustomData(
            age=float(request.form.get("age")),
            sex=int(request.form.get("sex")),
            cp=int(request.form.get("cp")),
            trestbps=float(request.form.get("trestbps")),
            chol=float(request.form.get("chol")),
            fbs=int(request.form.get("fbs")),
            restecg=int(request.form.get("restecg")),
            thalch=float(request.form.get("thalch")),
            exang=int(request.form.get("exang")),
            oldpeak=float(request.form.get("oldpeak")),
            slope=int(request.form.get("slope")),
            ca=int(request.form.get("ca")),
            thal=int(request.form.get("thal"))
        )

        pred_df = data.get_data_as_dataframe()

        pipeline = PredictPipeline()
        result = pipeline.predict(pred_df)

        return render_template(
            "index.html",
            prediction=result["prediction"],
            probability=result["probability"],
            risk=result["risk_level"],
            suggestions=result["suggestions"]
        )

    except Exception as e:
        return str(e)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)