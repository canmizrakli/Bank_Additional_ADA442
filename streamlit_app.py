# Import necessary libraries
import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Title and description of the app
st.title("Bank Term Deposit Prediction")
st.write("Predict whether a client will subscribe to a term deposit based on provided features. Created by Ataberk Kılavuzcu, Bora Kutun, Can Mızraklı and Umut Ulaş Balcı.")

# Load the trained model
try:
    model = pickle.load(open('final_model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Model file 'final_model.pkl' not found. Please ensure the model is in the same directory as this script.")
    st.stop()

# Load feature names
try:
    with open('feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
except FileNotFoundError:
    st.error("Feature names file 'feature_names.pkl' not found. Please ensure it is in the same directory as this script.")
    st.stop()

# Input fields for user-provided data
st.header("Input Client Features")
age = st.number_input("Age", min_value=18, max_value=100, value=30)
job = st.selectbox("Job", ["admin.", "technician", "services", "management", "retired", "blue-collar", 
                           "unemployed", "unknown", "self-employed", "entrepreneur", "housemaid", "student"])
marital = st.selectbox("Marital Status", ["married", "single", "divorced"])
education = st.selectbox("Education", ["unknown", "secondary", "primary", "tertiary"])
default = st.selectbox("Has Credit in Default?", ["no", "yes"])
balance = st.number_input("Balance (in euros)", min_value=-2000, max_value=100000, value=0)
housing = st.selectbox("Has Housing Loan?", ["no", "yes"])
loan = st.selectbox("Has Personal Loan?", ["no", "yes"])
contact = st.selectbox("Contact Communication Type", ["unknown", "cellular", "telephone"])
day = st.number_input("Day of the Month (Last Contact)", min_value=1, max_value=31, value=15)
month = st.selectbox("Month of Last Contact", ["jan", "feb", "mar", "apr", "may", "jun", 
                                                "jul", "aug", "sep", "oct", "nov", "dec"])
duration = st.number_input("Duration of Last Contact (seconds)", min_value=0, max_value=5000, value=0)
campaign = st.number_input("Number of Contacts During Campaign", min_value=1, max_value=50, value=1)
pdays = st.number_input("Days Since Client Was Last Contacted (-1 means never)", min_value=-1, max_value=999, value=-1)
previous = st.number_input("Number of Contacts Before this Campaign", min_value=0, max_value=50, value=0)
poutcome = st.selectbox("Outcome of Previous Campaign", ["unknown", "success", "failure", "other"])

# Create a dictionary for user input
input_dict = {
    "age": age,
    "job": job,
    "marital": marital,
    "education": education,
    "default": default,
    "balance": balance,
    "housing": housing,
    "loan": loan,
    "contact": contact,
    "day": day,
    "month": month,
    "duration": duration,
    "campaign": campaign,
    "pdays": pdays,
    "previous": previous,
    "poutcome": poutcome
}

# Convert user input into a DataFrame for preprocessing
input_df = pd.DataFrame([input_dict])

# Apply one-hot encoding to match training preprocessing
input_df = pd.get_dummies(input_df)

# Reorder the input DataFrame to match the model's training feature order
input_df = input_df.reindex(columns=feature_names, fill_value=0)

# Prediction button
if st.button("Predict"):
    try:
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)

        # Display results
        if prediction[0] == 1:
            st.success("The client is likely to subscribe to the term deposit.")
        else:
            st.warning("The client is not likely to subscribe to the term deposit.")

        st.write(f"Prediction probabilities: {prediction_proba}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")