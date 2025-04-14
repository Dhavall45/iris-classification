import streamlit as st
import numpy as np
import joblib

# Load the saved model and scaler
model = joblib.load("iris_classifier_model.pkl")
scaler = joblib.load("iris_scaler.pkl")
class_names = ['Setosa', 'Versicolor', 'Virginica']

st.title("🌸 Iris Flower Classifier")
st.write("Enter the flower's measurements to predict its species.")

# Input sliders
sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 0.2)

# Prepare the input
features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
features_scaled = scaler.transform(features)

# Predict
if st.button("Predict"):
    prediction = model.predict(features_scaled)[0]
    predicted_class = class_names[prediction]
    st.success(f"The predicted species is: **{predicted_class}**")
