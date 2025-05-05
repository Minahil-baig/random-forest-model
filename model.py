import streamlit as st
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from streamlit import session_state

    # Load trained model
model = joblib.load('rf_model_trained.joblib')
st.title("Water Quality Classifier")

# Initialize session state
if "submitted" not in st.session_state:
    st.session_state.submitted = False

# Input fields
ph = st.number_input("pH", min_value=0.0, max_value=14.0)
hardness = st.number_input("Hardness")
solids = st.number_input("Solids")
chloramines = st.number_input("Chloramines")
sulfate = st.number_input("Sulfate")
conductivity = st.number_input("Conductivity")
organic_carbon = st.number_input("Organic Carbon")
trihalomethanes = st.number_input("Trihalomethanes")
turbidity = st.number_input("Turbidity")

# Submit button with session state
if st.button("Classify"):
    st.session_state.submitted = True
    features = np.array([[ph, hardness, solids, chloramines, sulfate, conductivity,
                          organic_carbon, trihalomethanes, turbidity]])
    prediction = model.predict(features)[0]
    st.session_state.prediction = "Safe" if prediction == 1 else "Unsafe"

# Show prediction if submitted
if st.session_state.submitted:
    st.success(f"Water is predicted to be: *{st.session_state.prediction}*")