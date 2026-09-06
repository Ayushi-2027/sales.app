import streamlit as st
import joblib
import pandas as pd

# Load the trained model
loaded_lr_model = joblib.load("linear_regression_model.joblib")

# Page title
st.title("Sales Prediction using Linear Regression")

st.write(
    "Enter the advertising spending for TV, Radio, and Newspaper "
    "to predict sales."
)

# Input features
tv = st.slider(
    "TV Advertising Spending ($)",
    min_value=0.0,
    max_value=300.0,
    value=150.0
)

radio = st.slider(
    "Radio Advertising Spending ($)",
    min_value=0.0,
    max_value=50.0,
    value=25.0
)

newspaper = st.slider(
    "Newspaper Advertising Spending ($)",
    min_value=0.0,
    max_value=120.0,
    value=30.0
)

# Create input DataFrame
input_data = pd.DataFrame({
    "TV": [tv],
    "Radio": [radio],
    "Newspaper": [newspaper]
})

# Predict sales
if st.button("Predict Sales"):

    prediction = loaded_lr_model.predict(input_data)

    st.success(
        f"Predicted Sales: {prediction[0]:.2f} units"
    )
