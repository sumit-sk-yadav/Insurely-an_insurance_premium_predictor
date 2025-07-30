import streamlit as st
import joblib
import pandas as pd
import shap
import numpy as np
import matplotlib.pyplot as plt
from model_utils import load_all, date, get_feature_names, make_prediction

st.markdown("""
    <style>
    /* Page background */
    .main {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 8px;
    }

    /* Title and section headers */
    h1, h2, h3 {
        color: #2c3e50;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Label text for widgets */
    .stTextInput > label, .stSelectbox > label, .stSlider > label, .stDateInput > label {
        font-weight: 600;
        color: #34495e;
        font-size: 0.95rem;
        padding-bottom: 0.25rem;
    }

    /* Custom section dividers */
    .block-container {
        padding-top: 1rem;
    }

    /* Add a subtle box-shadow to each form section */
    section[data-testid="stHorizontalBlock"] {
        background-color: #ffffff;
        padding: 1.5rem 1rem;
        margin-bottom: 1.2rem;
        border-radius: 8px;
        box-shadow: 0px 2px 6px rgba(0, 0, 0, 0.05);
    }

    /* Buttons */
    div.stButton > button {
        background-color: #2e86de;
        color: white;
        font-weight: 600;
        border-radius: 6px;
        padding: 0.6rem 1rem;
        border: none;
    }

    div.stButton > button:hover {
        background-color: #1e5fa2;
        transition: background-color 0.3s ease;
    }

    /* Adjust widget spacing */
    .stSelectbox, .stSlider, .stTextInput, .stNumberInput, .stDateInput {
        margin-bottom: 1.2rem;
    }

    /* SHAP plot */
    .element-container:has(.shap) {
        margin-top: 2rem;
        padding: 1rem;
        background-color: #ecf0f1;
        border-radius: 8px;
    }
    </style>
""", unsafe_allow_html=True)


# Streamlit app config
st.set_page_config(page_title="Insurely", layout="centered")
st.title("Insurely")
st.subheader("an insurance premium predictor trained over premium details of over a million individuals")
st.markdown("---")

# Load model components
model, preprocessor, explainer, numerical_features, categorical_features = load_all()

# User Input Section
st.header("Customer Details")
st.markdown("#### Personal Info")
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 18, 100, 30, key="age")
    gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"], key="marital")
    dependents = st.slider("Number of Dependents", 0, 10, 0, key="dependents")
with col2:
    education = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"], key="education")
    occupation = st.selectbox("Occupation", ["Self-employed", "Unemployed", "Employed"], key="occupation")
    location = st.selectbox("Location", ["Urban", "Rural", "Suburban"], key="location")
    income = st.number_input("Annual Income", 5000, 200000, 30000, key="income")


st.markdown("#### Lifestyle & Health")
col3, col4 = st.columns(2)
with col3:
    health_score = st.slider("Health Score", 0.0, 100.0, 50.0, key="health")
    smoking = st.selectbox("Smoking Status", ["Yes", "No"], key="smoking")
with col4:
    exercise = st.selectbox("Exercise Frequency", ["Rarely", "Monthly", "Weekly", "Daily"], key="exercise")
    feedback = st.selectbox("Customer Feedback", ["Poor", "Average", "Good"], key="feedback")


st.markdown("#### Policy Info")
col5, col6 = st.columns(2)
with col5:
    policy_type = st.selectbox("Policy Type", ["Premium", "Comprehensive", "Basic"], key="policy_type")
    insurance_duration = st.slider("Insurance Duration (years)", 0, 20, 2, key="ins_duration")
with col6:
    previous_claims = st.slider("Number of Previous Claims", 0, 10, 0, key="prev_claims")
    policy_start_date = st.date_input("Policy Start Date", value=pd.to_datetime("2022-01-01"), key="policy_date")


st.markdown("#### Property Information")
col7, col8 = st.columns(2)
with col7:
    vehicle_age = st.slider("Vehicle Age (years)", 0, 30, 5, key="vehicle_age")
with col8:
    credit_score = st.slider("Credit Score", 300, 850, 600, key="credit_score")
    property_type = st.selectbox("Property Type", ["House", "Apartment", "Condo"], key="property")


# Create input DataFrame
input_dict = {
    'Age': age,
    'Gender': gender,
    'Annual Income': income,
    'Marital Status': marital_status,
    'Number of Dependents': dependents,
    'Education Level': education,
    'Health Score': health_score,
    'Location': location,
    'Policy Type': policy_type,
    'Vehicle Age': vehicle_age,
    'Credit Score': credit_score,
    'Insurance Duration': insurance_duration,
    'Customer Feedback': feedback,
    'Smoking Status': smoking,
    'Exercise Frequency': exercise,
    'Property Type': property_type,
    'Occupation': occupation,
    'Previous Claims': previous_claims,
    'Policy Start Date': pd.to_datetime(policy_start_date)
}

input_df = pd.DataFrame([input_dict])
input_df = date(input_df)
drop_cols = ['id', 'Group', 'Year', 'Month', 'Day', 'Week']
input_df.drop(columns=[col for col in drop_cols if col in input_df.columns], inplace=True)




# Prediction and SHAP explanation
if st.button("Predict Premium"):
    try:
        prediction, X_transformed = make_prediction(input_df, preprocessor, model)
        st.success(f"Predicted Insurance Premium: {prediction:.2f}")

        st.subheader("SHAP Explanation")
        st.subheader("biggest effectors on your premium amount")

        # Reconstruct feature names
        all_feature_names = get_feature_names(preprocessor, numerical_features, categorical_features)
        X_explainer = pd.DataFrame(X_transformed, columns=all_feature_names)

        # Compute SHAP values
        shap_explanation = explainer(X_explainer)

        # Waterfall plot
        fig, ax = plt.subplots(figsize=(10, 5))
        shap.plots.waterfall(shap_explanation[0], max_display=15, show=False)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"An error occurred during prediction or SHAP explanation: {e}")

