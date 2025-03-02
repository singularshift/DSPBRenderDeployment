import joblib
import xgboost as xgb
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import os

# Function to Load XGBoost Model with Pickle (Ensures XGBRegressor, not Booster)
def load_xgboost_model():
    try:
        # First try loading with joblib
        model = joblib.load("best_xgboost_model.pkl")
        
        # If it's a booster, convert it to XGBRegressor
        if isinstance(model, xgb.Booster):
            print("Converting Booster to XGBRegressor...")
            # Save the booster to a temporary file
            model.save_model("temp_model.json")
            # Create new XGBRegressor and load the model
            xgb_regressor = xgb.XGBRegressor()
            xgb_regressor.load_model("temp_model.json")
            # Clean up temporary file
            os.remove("temp_model.json")
            return xgb_regressor
        
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

# Load the trained XGBoost model
model = load_xgboost_model()
print("XGBoost model loaded successfully!")

# Initialize FastAPI app
app = FastAPI(title="Car Price Prediction API", description="Predict car prices using a trained XGBoost model.", version="1.0")

# Root Endpoint (Fixes 404 Not Found Error)
@app.get("/")
def home():
    return {"message": "Welcome to the Car Price Prediction API! Use /predict to get predictions."}

# Define Input Data Model
class CarFeatures(BaseModel):
    prod_year: int
    engine_volume: float
    mileage: int
    cylinders: int
    airbags: int
    turbo: int

# Prediction Endpoint
@app.post("/predict")
def predict_price(car_features: CarFeatures):
    input_data = pd.DataFrame([[
        car_features.prod_year,
        car_features.engine_volume,
        car_features.mileage,
        car_features.cylinders,
        car_features.airbags,
        car_features.turbo
    ]], columns=['prod_year', 'engine_volume', 'mileage', 'cylinders', 'airbags', 'turbo'])
    
    predicted_price = float(model.predict(input_data)[0])  # Convert numpy.float32 to Python float
    return {"predicted_price": predicted_price}
