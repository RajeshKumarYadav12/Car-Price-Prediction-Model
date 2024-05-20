import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('random_forest_regression_model.pkl', 'rb'))

def preprocess_input(Year, Present_Price, Kms_Driven, Owner, Fuel_Type_Petrol, Seller_Type_Individual, Transmission_Manual):
    Fuel_Type_Diesel = 0
    Kms_Driven2 = np.log(Kms_Driven)
    Year = 2024 - Year

    if Fuel_Type_Petrol == 'Petrol':
        Fuel_Type_Petrol = 1
        Fuel_Type_Diesel = 0
    else:
        Fuel_Type_Petrol = 0
        Fuel_Type_Diesel = 1

    if Seller_Type_Individual == 'Individual':
        Seller_Type_Individual = 1
    else:
        Seller_Type_Individual = 0

    if Transmission_Manual == 'Automatic':
        Transmission_Manual = 0
    else:
        Transmission_Manual = 1

    return [[Present_Price, Kms_Driven2, Owner, Year, Fuel_Type_Diesel, Fuel_Type_Petrol, Seller_Type_Individual, Transmission_Manual]]

def main():
    st.title('Car Price Prediction')
    st.write('Enter Car Details:', unsafe_allow_html=True)
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
    if st.button('Predict'):
        input_data = preprocess_input(Year, Present_Price, Kms_Driven, Owner, Fuel_Type_Petrol, Seller_Type_Individual, Transmission_Manual)
        prediction = model.predict(input_data)
        output = round(prediction[0], 2)
        if output < 0:
            st.error("Sorry, you cannot sell this car.")
        else:
            st.success("You can sell the car at {}".format(output))

if __name__ == '__main__':
    main()
