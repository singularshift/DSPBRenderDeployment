import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

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

# Prediction Endpoint
@app.post("/predict")
async def predict(features: CarFeatures):
    try:
        model = joblib.load("best_gradient_boosting_model.pkl")
        print("Best Gradient Boosting model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        model = None

    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Create input array in correct order
        input_data = [[
            features.prod_year,
            features.engine_volume,
            features.mileage,
            features.cylinders,
            features.airbags,
            features.turbo
        ]]
        
        prediction = model.predict(input_data)
        return {"predicted_price": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app_gbr:app", host="127.0.0.1", port=8000, reload=True)
