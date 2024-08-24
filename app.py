import streamlit as st
import pickle
import numpy as np
import pandas as pd
import warnings

# Suppress warnings from scikit-learn
warnings.filterwarnings("ignore", category=Warning)

# Load the trained model with error handling
try:
    with open('random_forest_regression_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
except Exception as e:
    st.error(f"Error loading model: {e}")
    model = None

def preprocess_input(Year, Present_Price, Kms_Driven, Owner, Fuel_Type_Petrol, Seller_Type_Individual, Transmission_Manual):
    Fuel_Type_Diesel = 0

    # Handle zero Kms_Driven separately to avoid taking logarithm of zero
    Kms_Driven2 = np.log1p(Kms_Driven)  # Use np.log1p for log(1 + x), avoids issues with zero

    Year = 2024 - Year

    Fuel_Type_Petrol = 1 if Fuel_Type_Petrol == 'Petrol' else 0
    Fuel_Type_Diesel = 1 - Fuel_Type_Petrol

    Seller_Type_Individual = 1 if Seller_Type_Individual == 'Individual' else 0
    Transmission_Manual = 0 if Transmission_Manual == 'Automatic' else 1

    return [[Present_Price, Kms_Driven2, Owner, Year, Fuel_Type_Diesel, Fuel_Type_Petrol, Seller_Type_Individual,
             Transmission_Manual]]

def main():
    st.title('Car Price Prediction')
    st.write('Enter Car Details:', unsafe_allow_html=True)

    # Input fields
    col1, col2 = st.columns(2)
    with col1:
        Year = st.number_input('Year', value=2021, step=1, format='%d')
        Present_Price = st.number_input('Present Price (in lakh)', value=5.0, step=0.1, format="%.1f")
        Kms_Driven = st.number_input('Kilometers Driven', value=50000, step=1000, format="%d")
    with col2:
        Owner = st.number_input('Owner', value=0, step=1, format='%d')
        Fuel_Type_Petrol = st.selectbox('Fuel Type', ['Petrol', 'Diesel'])
        Seller_Type_Individual = st.selectbox('Seller Type', ['Individual', 'Dealer'])
        Transmission_Manual = st.selectbox('Transmission', ['Manual', 'Automatic'])

    # Prediction
    if st.button('Predict'):
        if model is not None:
            input_data = preprocess_input(Year, Present_Price, Kms_Driven, Owner, Fuel_Type_Petrol, Seller_Type_Individual,
                                          Transmission_Manual)
            prediction = model.predict(input_data)
            output = round(prediction[0], 2)
            if output < 0:
                st.error("Sorry, you cannot sell this car.")
            else:
                st.success(f"You can sell the car at {output} lakh")
        else:
            st.error("Model could not be loaded. Please check the model file.")

if __name__ == '__main__':
    main()
