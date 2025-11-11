from fastapi import FastAPI, Query
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# Load trained model
if not os.path.exists("model2.pkl"):
    raise FileNotFoundError("model.pkl not found. Run Regression.py first!")

model = joblib.load("model2.pkl")
app = FastAPI(title="House Price Predictor", version="1.0.0")

class PredictPayload(BaseModel):
    square_footage: float
    num_bedrooms: int
    num_bathrooms: int
    year_built: int
    lot_size: float
    garage_size: int
    neighborhood_quality: int

    def to_dataframe(self):
        data = {
            "Square_Footage": [self.square_footage],
            "Num_Bedrooms": [self.num_bedrooms],
            "Num_Bathrooms": [self.num_bathrooms],
            "Year_Built": [self.year_built],
            "Lot_Size": [self.lot_size],
            "Garage_Size": [self.garage_size],
            "Neighborhood_Quality": [self.neighborhood_quality],
        }
        return pd.DataFrame(data)

@app.get("/predict")
def predict_get(
    square_footage: float = Query(...),
    num_bedrooms: int = Query(...),
    num_bathrooms: int = Query(...),
    year_built: int = Query(...),
    lot_size: float = Query(...),
    garage_size: int = Query(...),
    neighborhood_quality: int = Query(...),
):
    print("Get Ok")
    payload = PredictPayload(
        square_footage=square_footage,
        num_bedrooms=num_bedrooms,
        num_bathrooms=num_bathrooms,
        year_built=year_built,
        lot_size=lot_size,
        garage_size=garage_size,
        neighborhood_quality=neighborhood_quality
    )
    print("payload")
    print(payload)
    

    df = payload.to_dataframe()
    print("df")
    print(df)

    pred = model.predict(df)[0]
    return {"predicted_price": float(pred)}

@app.post("/predict")
def predict_post(payload: PredictPayload):
    df = payload.to_dataframe()
    pred = model.predict(df)[0]
    return {"predicted_price": float(pred)}

@app.get("/health")
def health():
    return {"status": "ok"}

