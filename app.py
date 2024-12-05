import streamlit as st
import joblib
import pandas as pd

# # Configure the Streamlit page
# st.set_page_config(
#     page_title="Healthcare Payment Prediction",
#     layout="wide",  # Expands the page layout to use the full width
#     initial_sidebar_state="collapsed"
# )

# Load the trained model and encoders
model = joblib.load('models/xgboost_20241204_170555.pkl')
encoders = joblib.load('models/encoders_20241204_170555.pkl')

# List of columns (exact order used during training)
model_feature_order = [
    "OPDATEYR", "OPDATEMM", "SEEDOC_M18", "DRSPLTY_M18", "MEDPTYPE_M18",
    "VSTCTGRY", "VSTRELCN_M18", "LABTEST_M18", "SONOGRAM_M18", "XRAYS_M18",
    "MAMMOG_M18", "MRI_M18", "EKG_M18", "RCVVAC_M18", "SURGPROC", "MEDPRESC",
    "VISITTYPE", "TELEHEALTHFLAG", "FFOPTYPE", "ICD10CDX", "icd_block", "visit_complexity", "icd_chapter"
]

st.title("Healthcare Payment Prediction")

# Display column order for user reference
st.write("Enter the values for all features as a comma-separated list in the exact order shown below:")
st.code(", ".join(model_feature_order))

# User input for real values as a comma-separated list
user_input = st.text_input("Enter values:")

# Predict button
if st.button("Predict"):
    try:
        # Convert the input values into a list
        input_list = [value.strip() for value in user_input.split(",")]

        # Check if the correct number of inputs is provided
        if len(input_list) != len(model_feature_order):
            st.error(f"Please provide exactly {len(model_feature_order)} values.")
        else:
            # Create a DataFrame from the input values
            input_df = pd.DataFrame([input_list], columns=model_feature_order)

            # Ensure correct data types
            for col in input_df.columns:
                if col in encoders:  # Categorical columns
                    input_df[col] = input_df[col].astype(str)
                else:  # Numeric columns
                    input_df[col] = pd.to_numeric(input_df[col], errors='coerce')

            # Encode the real values using encoders
            for col, encoder in encoders.items():
                if col in input_df.columns:
                    real_values = input_df[col].astype(str)
                    try:
                        input_df[col] = encoder.transform(real_values)
                    except ValueError:
                        st.error(f"Value '{real_values.iloc[0]}' in column '{col}' is not recognized. Mapping to 'Unknown'.")
                        input_df[col] = encoder.transform(["Unknown"])

            # Ensure all features are present and ordered as expected by the model
            input_df = input_df[model.get_booster().feature_names]

            # Make prediction
            prediction = model.predict(input_df)[0]
            st.success(f"Predicted Payment: ${prediction:.2f}")

    except Exception as e:
        st.error(f"Error processing input: {e}")
