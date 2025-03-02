import streamlit as st
import requests
import json
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="centered"
)

# Add title and description
st.title("🚗 Car Price Predictor")
st.write("Enter your car's details below to get an estimated price prediction!")

# Create input form
with st.form("prediction_form"):
    # Production Year
    current_year = datetime.now().year
    prod_year = st.slider(
        "Production Year",
        min_value=1950,
        max_value=current_year,
        value=2015,
        help="Select the year your car was manufactured"
    )

    # Engine Volume
    engine_volume = st.number_input(
        "Engine Volume (L)",
        min_value=0.1,
        max_value=10.0,
        value=2.0,
        step=0.1,
        help="Enter the engine volume in liters"
    )

    # Mileage
    mileage = st.number_input(
        "Mileage (km)",
        min_value=0,
        max_value=1000000,
        value=50000,
        step=1000,
        help="Enter the total kilometers driven"
    )

    # Create two columns for the remaining inputs
    col1, col2 = st.columns(2)

    with col1:
        # Cylinders
        cylinders = st.selectbox(
            "Number of Cylinders",
            options=[3, 4, 5, 6, 8, 10, 12],
            index=1,
            help="Select the number of cylinders in your car's engine"
        )

        # Airbags
        airbags = st.number_input(
            "Number of Airbags",
            min_value=0,
            max_value=12,
            value=2,
            step=1,
            help="Enter the number of airbags in your car"
        )

    with col2:
        # Turbo
        turbo = st.selectbox(
            "Turbo Engine",
            options=["No", "Yes"],
            index=0,
            help="Select whether your car has a turbo engine"
        )
        # Convert turbo to binary
        turbo = 1 if turbo == "Yes" else 0

    # Submit button
    submit_button = st.form_submit_button("Predict Price")

# Handle form submission
if submit_button:
    # Prepare the input data
    input_data = {
        "prod_year": int(prod_year),  # Ensure integer
        "engine_volume": float(engine_volume),  # Ensure float
        "mileage": int(mileage),  # Ensure integer
        "cylinders": int(cylinders),  # Ensure integer
        "airbags": int(airbags),  # Ensure integer
        "turbo": int(turbo)  # Ensure integer
    }

    try:
        # Make prediction request to FastAPI backend
        response = requests.post(
            "http://localhost:8000/predict",
            json=input_data,
            headers={"accept": "application/json"}
        )
        
        try:
            # First try to parse as JSON
            result = response.json()
            prediction = float(result["predicted_price"])
        except (json.JSONDecodeError, ValueError, TypeError):
            # If JSON parsing fails, try to get the raw text and extract the number
            import re
            text = response.text
            # Look for a number in the response
            match = re.search(r'[-+]?\d*\.\d+|\d+', text)
            if match:
                prediction = float(match.group())
            else:
                raise ValueError("Could not extract prediction value from response")
        
        # Display the prediction with formatting
        st.success("Prediction Complete!")
        st.metric(
            label="Predicted Car Price",
            value=f"${prediction:,.2f}"
        )
    
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the prediction service. Please make sure the FastAPI server is running.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        if hasattr(response, 'text'):
            st.error(f"Server response: {response.text}")

# Add information about the model
with st.expander("ℹ️ About this app"):
    st.write("""
    This app predicts car prices based on various features using a trained XGBoost model. 
    The prediction is made in real-time using a FastAPI backend service.
    
    To use the app:
    1. Fill in your car's details in the form above
    2. Click the 'Predict Price' button
    3. Get an instant price prediction for your car
    
    Note: Make sure the FastAPI backend server is running before making predictions.
    """)
