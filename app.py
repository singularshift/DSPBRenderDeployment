import joblib
import xgboost as xgb
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# ✅ Function to Load XGBoost Model with Pickle (Ensures XGBRegressor, not Booster)
def load_xgboost_model():
    model = joblib.load("best_xgboost_model.pkl")

    # Ensure it's an XGBRegressor, not a Booster
    if isinstance(model, xgb.Booster):
        print("⚠️ Warning: Loaded model is a Booster. Converting to XGBRegressor.")
        xgb_regressor = xgb.XGBRegressor()
        xgb_regressor.load_model("best_xgboost_model.json")  # Convert Booster to XGBRegressor
        return xgb_regressor
    
    return model

# Load the trained XGBoost model
model = load_xgboost_model()
print("✅ XGBoost model loaded successfully!")

# Initialize FastAPI app
app = FastAPI(title="Car Price Prediction API", description="Predict car prices using a trained XGBoost model.", version="1.0")

# ✅ Root Endpoint (Fixes 404 Not Found Error)
@app.get("/")
def home():
    return {"message": "Welcome to the Car Price Prediction API! Use /predict to get predictions."}

# ✅ Define Input Data Model
class CarFeatures(BaseModel):
    turbo: int
    airbags: int
    prod_year: int
    cylinders: int
    engine_volume: float
    mileage: int

# ✅ Prediction Endpoint
@app.post("/predict")
def predict_price(features: CarFeatures):
    # Convert input to DataFrame
    input_data = pd.DataFrame([features.dict()])

    # Make prediction
    predicted_price = model.predict(input_data)[0]

    return {"predicted_price": predicted_price}
