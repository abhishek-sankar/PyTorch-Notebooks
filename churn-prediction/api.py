from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()


# Load your model at startup
# Make sure to place "best_xgb_model.pkl" or your ensemble .pkl in the same dir
@app.on_event("startup")
def load_model():
    global model
    model = joblib.load("best_xgb_model.pkl")


class CustomerData(BaseModel):
    features: list


@app.post("/predict")
def predict_churn(data: CustomerData):
    array_data = np.array(data.features).reshape(1, -1)
    prediction = model.predict(array_data)
    return {"prediction": int(prediction[0])}
