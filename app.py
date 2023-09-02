import streamlit as st
import numpy as np
import pandas as pd
import joblib  # Import joblib directly

# Load the saved model
model = joblib.load("C:/Users/Youssef/Desktop/Copy_of_Rock_vs_Mine_Prediction/Rock_vs_Mine_Prediction.sav")

# Function to make predictions
def predict(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = model.predict(input_data_reshaped)
    return prediction[0]

# Streamlit app
def main():
    st.title("Rock vs. Mine Prediction")
    st.sidebar.header("User Input")

    # Create input fields for user to enter data
    input_data = []
    for i in range(60):
        input_data.append(st.sidebar.number_input(f"Feature {i + 1}", value=0.0))

    if st.sidebar.button("Predict"):
        prediction = predict(input_data)
        if prediction == 'R':
            st.write("The object is a Rock")
        else:
            st.write("The object is a Mine")

if __name__ == "__main__":
    main()

