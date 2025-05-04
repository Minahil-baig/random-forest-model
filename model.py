import streamlit as st
import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier

# Load trained model
model = joblib.load('rf_model_trained.joblib')

st.title("Water Quality Classification")

st.write("Enter the water sample features below:")

# Input fields
ph = st.number_input("pH", 0.0, 14.0)
hardness = st.number_input("Hardness", 0.0)
solids = st.number_input("Solids", 0.0)
chloramines = st.number_input("Chloramines", 0.0)
sulfate = st.number_input("Sulfate", 0.0)
conductivity = st.number_input("Conductivity", 0.0)
organic_carbon = st.number_input("Organic Carbon", 0.0)
trihalomethanes = st.number_input("Trihalomethanes", 0.0)
turbidity = st.number_input("Turbidity", 0.0)

# Predict
if st.button("Check Water Quality"):
    input_data = pd.DataFrame([[ph, hardness, solids, chloramines, sulfate, conductivity,
                                organic_carbon, trihalomethanes, turbidity]],
                              columns=["ph", "Hardness", "Solids", "Chloramines", "Sulfate",
                                       "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"])

    prediction = model.predict(input_data)[0]
    result = "Safe for Drinking" if prediction == 1 else "Not Safe for Drinking"
    st.success(f"Prediction:{result}")
