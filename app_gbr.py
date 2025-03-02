import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn

# Load the trained Gradient Boosting model
model = joblib.load("best_gradient_boosting_model.pkl")
print("Best Gradient Boosting model loaded successfully!")

# Initialize FastAPI app
app = FastAPI(title="Car Price Prediction API (Gradient Boosting)", description="Predict car prices using an optimized Gradient Boosting model.")

# Define input data structure
class CarFeatures(BaseModel):
    turbo: int
    airbags: int
    prod_year: int
    cylinders: int
    engine_volume: float
    mileage: int

# Root Endpoint (Fixes 404 Not Found)
@app.get("/")
def home():
    return {"message": "Welcome to the Car Price Prediction API! Use /predict to get predictions."}

# ✅ Prediction Endpoint
@app.post("/predict")
def predict_price(features: CarFeatures):
    # Convert input to DataFrame
    input_data = pd.DataFrame([features.dict()])

    # Predict price
    predicted_price = model.predict(input_data)[0]

    return {"predicted_price": predicted_price}

if __name__ == "__main__":
    uvicorn.run("app_gbr:app", host="127.0.0.1", port=8000, reload=True)
