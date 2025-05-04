# # # import streamlit as st
# # # import pandas as pd
# # # import numpy as np
# # # import joblib
# # #
# # # # Load the model
# # # model = joblib.load('water_quality_model.pkl')
# # #
# # # st.title("Water Quality ")
# # # st.write("Enter the water sample characteristics below:")
# # #
# # # # Example features (adjust based on your dataset)
# # # ph = st.number_input("pH", min_value=0.0, max_value=14.0, step=0.1)
# # # Hardness = st.number_input("Hardness", min_value=0.0)
# # # Solids = st.number_input("Solids", min_value=0.0)
# # # Chloramines = st.number_input("Chloramines", min_value=0.0)
# # # Sulfate = st.number_input("Sulfate", min_value=0.0)
# # # Conductivity = st.number_input("Conductivity", min_value=0.0)
# # # Organic_carbon = st.number_input("Organic Carbon", min_value=0.0)
# # # Trihalomethanes = st.number_input("Trihalomethanes", min_value=0.0)
# # # Turbidity = st.number_input("Turbidity", min_value=0.0)
# # #
# # # # Create input array
# # # input_data = np.array([[ph, Hardness, Solids, Chloramines, Sulfate,
# # #                         Conductivity, Organic_carbon, Trihalomethanes, Turbidity]])
# # #
# # # # Predict
# # # if st.button("Classify Water Quality"):
# # #     prediction = model.predict(input_data)
# # #     result = "Safe to Drink" if prediction[0] == 1 else "Not Safe to Drink"
# # #     st.success(f"Prediction: {result}")
# #
# #
# #
# #
# #
# #
# #
# # import streamlit as st
# # import pandas as pd
# # import joblib
# #
# # # Load the model
# # model = joblib.load("water_model.pkl")
# #
# # st.title("Water Quality Classification App")
# #
# # # Input fields
# # ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0)
# # hardness = st.number_input("Hardness", value=150.0)
# # solids = st.number_input("Solids", value=10000.0)
# # chloramines = st.number_input("Chloramines", value=7.0)
# # sulfate = st.number_input("Sulfate", value=300.0)
# # conductivity = st.number_input("Conductivity", value=400.0)
# # organic_carbon = st.number_input("Organic Carbon", value=10.0)
# # trihalomethanes = st.number_input("Trihalomethanes", value=80.0)
# # turbidity = st.number_input("Turbidity", value=3.0)
# #
# # # Predict button
# # if st.button("Classify Water Quality"):
# #     input_data = pd.DataFrame([[ph, hardness, solids, chloramines, sulfate,
# #                                 conductivity, organic_carbon, trihalomethanes, turbidity]],
# #                               columns=["ph", "Hardness", "Solids", "Chloramines", "Sulfate",
# #                                        "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"])
# #     prediction = model.predict(input_data)
# #     if prediction[0] == 1:
# #         st.success("The water is *Safe* to drink.")
# #     else:
# #         st.error("The water is *Not Safe* to drink.")
#
#
#
#
#
#
# import streamlit as st
# import joblib
# import numpy as np
#
# # Load the trained model
# model8 = joblib.load('water_model.pkl')
#
# # Streamlit UI
# st.title("Water Quality Classification")
# st.write("This app classifies whether a water sample is safe or not.")
#
# # Take user input
# pH = st.number_input("pH Level")
# Hardness = st.number_input("Hardness")
# Solids = st.number_input("Solids")
# Chloramines = st.number_input("Chloramines")
# Sulfate = st.number_input("Sulfate")
# Conductivity = st.number_input("Conductivity")
# Organic_carbon = st.number_input("Organic Carbon")
# Trihalomethanes = st.number_input("Trihalomethanes")
# Turbidity = st.number_input("Turbidity")
#
# # Prediction
# if st.button("Predict"):
#     features = np.array([[pH, Hardness, Solids, Chloramines, Sulfate,
#                           Conductivity, Organic_carbon, Trihalomethanes, Turbidity]])
#     prediction = model.predict(features)
#     result = "Safe" if prediction[0] == 1 else "Not Safe"
#     st.success(f"The water is: {result}")

#
#
#
#
#
#
#
#
#
# impotrt sreamlit as st
# import joblib
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
# # Load model
# model=joblib.load('random_forest_model.joblib')
#
# st.title("💧 Water Quality Classifier")
# st.markdown("This app classifies a water sample as **Safe** or **Unsafe** to drink.")
#
# # User inputs
# ph = st.number_input("pH", 0.0, 14.0, step=0.1)
# hardness = st.number_input("Hardness", 0.0)
# solids = st.number_input("Solids", 0.0)
# chloramines = st.number_input("Chloramines", 0.0)
# sulfate = st.number_input("Sulfate", 0.0)
# conductivity = st.number_input("Conductivity", 0.0)
# organic_carbon = st.number_input("Organic Carbon", 0.0)
# trihalomethanes = st.number_input("Trihalomethanes", 0.0)
# turbidity = st.number_input("Turbidity", 0.0)
#
# # Predict
# if st.button("Classify"):
#     features = np.array([[ph, hardness, solids, chloramines, sulfate,
#                           conductivity, organic_carbon, trihalomethanes, turbidity]])
#     prediction = model.predict(features)[0]
#     result = "Safe" if prediction == 1 else "Unsafe"
#     st.success(f"Prediction: The water is **{result}** to drink.")






# import sreamlit as st
# import joblib
# import numpy as np
#
# from sklearn.ensemble import RandomForestClassifier
#
# # Load model
# model = joblib.load('random_forest_model.joblib')
# st.title("Water Quality Classification")
# st.write("Enter the water sample features to predict its potability (safe/unsafe).")
#
# # Input fields
# ph = st.number_input("pH", min_value=0.0, max_value=14.0, step=0.1)
# hardness = st.number_input("Hardness", min_value=0.0)
# solids = st.number_input("Solids", min_value=0.0)
# chloramines = st.number_input("Chloramines", min_value=0.0)
# sulfate = st.number_input("Sulfate", min_value=0.0)
# conductivity = st.number_input("Conductivity", min_value=0.0)
# organic_carbon = st.number_input("Organic Carbon", min_value=0.0)
# trihalomethanes = st.number_input("Trihalomethanes", min_value=0.0)
# turbidity = st.number_input("Turbidity", min_value=0.0)
#
# # Prediction
# if st.button("Classify"):
#     features = np.array([[ph, hardness, solids, chloramines, sulfate,
#                           conductivity, organic_carbon, trihalomethanes, turbidity]])
# prediction = model.predict(features)[0]
# result = "Safe to Drink" if prediction == 1
# else "Not Safe to Drink"
#     st.success(f"Result: {result}")







# import streamlit as st
#
# st.title("Water Quality Prediction App")
# st.write("Enter the parameters below to predict if the water is safe to drink.")
#
# # Input fields for each feature
# ph = st.number_input('pH')
# hardness = st.number_input('Hardness')
# solids = st.number_input('Solids')
# chloramines = st.number_input('Chloramines')
# sulfate = st.number_input('Sulfate')
# conductivity = st.number_input('Conductivity')
# organic_carbon = st.number_input('Organic Carbon')
# trihalomethanes = st.number_input('Trihalomethanes')
# turbidity = st.number_input('Turbidity')
#
# # Predict button
# if st.button('Predict'):
#     input_data = np.array([[ph, hardness, solids, chloramines, sulfate,
#                             conductivity, organic_carbon, trihalomethanes, turbidity]])
#     prediction = model.predict(input_data)[0]
#     result = "Safe to drink" if prediction == 1 else "Not safe to drink"
#     st.success(f"The water is **{result}**.")
#
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
#
# st.title("Water Quality Prediction App")
# st.write("Enter the parameters below to predict if the water is safe to drink.")
#
# # Input fields for each feature
# ph = st.number_input('pH')
# hardness = st.number_input('Hardness')
# solids = st.number_input('Solids')
# chloramines = st.number_input('Chloramines')
# sulfate = st.number_input('Sulfate')
# conductivity = st.number_input('Conductivity')
# organic_carbon = st.number_input('Organic Carbon')
# trihalomethanes = st.number_input('Trihalomethanes')
# turbidity = st.number_input('Turbidity')
#
# # Predict button
# if st.button('Predict'):
#     input_data = np.array([[ph, hardness, solids, chloramines, sulfate,
#                             conductivity, organic_carbon, trihalomethanes, turbidity]])
#     prediction = model.predict(input_data)[0]
#     result = "Safe to drink" if prediction == 1 else "Not safe to drink"
#     st.success(f"The water is **{result}**.")





# import streamlit as st
# import joblib
# import numpy as np
#
# from sklearn.ensemble import RandomForestClassifier
#
#
# # Load the trained model
# model = joblib.load('random_forest_model.joblib')
#
# st.title("Water Quality Prediction App")
# st.write("Enter the parameters below to predict if the water is safe to drink.")
#
# # Input fields for each feature
# ph = st.number_input('pH')
# hardness = st.number_input('Hardness')
# solids = st.number_input('Solids')
# chloramines = st.number_input('Chloramines')
# sulfate = st.number_input('Sulfate')
# conductivity = st.number_input('Conductivity')
# organic_carbon = st.number_input('Organic Carbon')
# trihalomethanes = st.number_input('Trihalomethanes')
# turbidity = st.number_input('Turbidity')
#
# # Predict button
# if st.button('Predict'):
#     input_data = np.array([[ph, hardness, solids, chloramines, sulfate,
#                             conductivity, organic_carbon, trihalomethanes, turbidity]])
#     prediction = model.predict(input_data)[0]
#     result = "Safe to drink" if prediction == 1 else "Not safe to drink"
#     st.success(f"The water is **{result}**.")








#
# import streamlit as st
# import joblib
# import numpy as np
# from sklearn.ensemble import RandomForestClassifier
#
# # Load trained model
# model = joblib.load('rf_model_trained.joblib')
#
# st.title("Water Quality Classification")
#
# st.markdown("Predict whether a water sample is **potable** or **not potable** based on its features.")
#
# # Feature input
# ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0)
# hardness = st.number_input("Hardness", min_value=0.0, value=100.0)
# solids = st.number_input("Solids", min_value=0.0, value=10000.0)
# chloramines = st.number_input("Chloramines", min_value=0.0, value=7.0)
# sulfate = st.number_input("Sulfate", min_value=0.0, value=300.0)
# conductivity = st.number_input("Conductivity", min_value=0.0, value=400.0)
# organic_carbon = st.number_input("Organic Carbon", min_value=0.0, value=10.0)
# trihalomethanes = st.number_input("Trihalomethanes", min_value=0.0, value=60.0)
# turbidity = st.number_input("Turbidity", min_value=0.0, value=4.0)
#
# # Prediction
# if st.button("Predict"):
#     input_features = np.array([[ph, hardness, solids, chloramines, sulfate,
#                                 conductivity, organic_carbon, trihalomethanes, turbidity]])
#     prediction = model.predict(input_features)[0]
#     result = "Potable" if prediction == 1 else "Not Potable"
#     st.success(f"The water sample is **{result}**.")








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