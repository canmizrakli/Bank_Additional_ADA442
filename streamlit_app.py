import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st

# App title
st.title("Bank Term Deposit Prediction")
st.write("Predict whether a client will subscribe to a term deposit based on provided features.")

# Check if the model file exists
model_path = 'final_model.pkl'

if not os.path.exists(model_path):
    st.error("Model file 'final_model.pkl' not found. Please upload it to the repository.")
    st.stop()

# Load the trained model
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Feature input form
st.sidebar.header("Input Features")

age = st.sidebar.number_input("Age", min_value=18, max_value=100, value=30)
job = st.sidebar.selectbox(
    "Job",
    options=[
        "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
        "retired", "self-employed", "services", "student", "technician", "unemployed", "unknown"
    ]
)
marital = st.sidebar.selectbox(
    "Marital Status",
    options=["married", "single", "divorced", "unknown"]
)
education = st.sidebar.selectbox(
    "Education",
    options=["basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate", "professional.course", "university.degree", "unknown"]
)
default = st.sidebar.selectbox(
    "Has Credit in Default?", options=["yes", "no", "unknown"]
)
housing = st.sidebar.selectbox(
    "Has Housing Loan?", options=["yes", "no", "unknown"]
)
loan = st.sidebar.selectbox(
    "Has Personal Loan?", options=["yes", "no", "unknown"]
)
contact = st.sidebar.selectbox(
    "Contact Communication Type", options=["cellular", "telephone", "unknown"]
)
month = st.sidebar.selectbox(
    "Last Contact Month",
    options=["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
)
day_of_week = st.sidebar.selectbox(
    "Last Contact Day of the Week", options=["mon", "tue", "wed", "thu", "fri"]
)
duration = st.sidebar.number_input("Duration of Last Contact (seconds)", min_value=0, max_value=5000, value=0)
campaign = st.sidebar.number_input("Number of Contacts in this Campaign", min_value=1, max_value=50, value=1)
pdays = st.sidebar.number_input("Days since Client was Last Contacted", min_value=-1, max_value=999, value=-1)
previous = st.sidebar.number_input("Number of Contacts Before this Campaign", min_value=0, max_value=100, value=0)
poutcome = st.sidebar.selectbox(
    "Outcome of Previous Campaign", options=["failure", "success", "nonexistent"]
)

# Input Data Preparation
input_data = pd.DataFrame({
    'age': [age],
    'job': [job],
    'marital': [marital],
    'education': [education],
    'default': [default],
    'housing': [housing],
    'loan': [loan],
    'contact': [contact],
    'month': [month],
    'day_of_week': [day_of_week],
    'duration': [duration],
    'campaign': [campaign],
    'pdays': [pdays],
    'previous': [previous],
    'poutcome': [poutcome]
})

# Ensure the features match the model's requirements
if os.path.exists("feature_names.pkl"):
    with open("feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    input_data = pd.get_dummies(input_data)
    input_data = input_data.reindex(columns=feature_names, fill_value=0)

# Make Prediction
if st.button("Predict"):
    try:
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        # Display results
        if prediction[0] == 1:
            st.success("The client is likely to subscribe to the term deposit.")
        else:
            st.warning("The client is unlikely to subscribe to the term deposit.")

        st.write(f"Prediction probabilities: {prediction_proba}")

    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")