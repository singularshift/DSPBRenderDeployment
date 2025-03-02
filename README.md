# Used Car Price Prediction API

This project implements machine learning models to predict used car prices based on vehicle features. The system uses Gradient Boosting and XGBoost regression models trained on car sales data and exposes predictions through a FastAPI interface.

## Project Overview

The project aims to provide accurate price predictions for used cars by analyzing key vehicle characteristics. The system leverages advanced regression techniques, feature engineering, and hyperparameter optimization to achieve high prediction accuracy.

### Key Features

- **Optimized Machine Learning Models**: Gradient Boosting Regressor and XGBoost models for accurate price prediction
- **RESTful API**: Easy-to-use FastAPI endpoints for real-time predictions
- **Data Preprocessing Pipeline**: Comprehensive cleaning and transformation strategies
- **Feature Selection**: Focused on 6 high-importance features for efficient predictions

## Usage

### Running the API Locally

You can run either the Gradient Boosting or XGBoost model API:

```bash
# Run the Gradient Boosting model API
python app_gbr.py

# Run the XGBoost model API
python app.py
```

The API will be available at <http://127.0.0.1:8000>

### API Endpoints

#### Home

- **URL**: `/`
- **Method**: `GET`
- **Description**: Welcome message and how to use the prediction endpoint

#### Price Prediction

- **URL**: `/predict`
- **Method**: `POST`
- **Request Body**:

```json
{
  "turbo": 1,
  "airbags": 6,
  "prod_year": 2018,
  "cylinders": 4,
  "engine_volume": 2.0,
  "mileage": 50000
}
```

- **Response**:

```json
{
  "predicted_price": 12500.75
}
```

### Example API Request with Python

```python
import requests

url = "http://127.0.0.1:8000/predict"
data = {
  "turbo": 1,
  "airbags": 6,
  "prod_year": 2018,
  "cylinders": 4,
  "engine_volume": 2.0,
  "mileage": 50000
}

response = requests.post(url, json=data)
print(response.json())
```

### Example API Request with cURL

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{
  "turbo": 1,
  "airbags": 6,
  "prod_year": 2018,
  "cylinders": 4,
  "engine_volume": 2.0,
  "mileage": 50000
}'
```

## Model Information

### Selected Features

After extensive analysis, six key features were selected for the model:

1. **prod_year**: Production year of the vehicle
2. **engine_volume**: Engine capacity in liters
3. **mileage**: Total distance traveled in kilometers
4. **cylinders**: Number of engine cylinders
5. **airbags**: Number of airbags
6. **turbo**: Binary indicator for turbocharged engines (0 = No, 1 = Yes)

## Deployment

The API is deployed on Render for evaluation by the professor.

## Future Improvements

- Incorporate additional vehicle features and historical pricing data
- Explore deep learning models for improved accuracy
- Add model interpretability using SHAP values
- Expand API functionalities with batch predictions and visualization endpoints
