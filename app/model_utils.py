# app/model_utils.py
import joblib
import shap
import os
import numpy as np
import pandas as pd

MODEL_PATH = "models/lgb_fin_model.joblib"
PREPROCESSOR_PATH = "models/preprocessor.joblib"
EXPLAINER_PATH = "models/shap_explainer.joblib"

def load_model():
    """Load trained LightGBM model."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

def load_preprocessor():
    """Load fitted preprocessing pipeline."""
    if not os.path.exists(PREPROCESSOR_PATH):
        raise FileNotFoundError(f"Preprocessor file not found at: {PREPROCESSOR_PATH}")
    return joblib.load(PREPROCESSOR_PATH)

def load_explainer():
    """Load pre-trained SHAP explainer."""
    if not os.path.exists(EXPLAINER_PATH):
        raise FileNotFoundError(f"SHAP explainer not found at: {EXPLAINER_PATH}")
    return joblib.load(EXPLAINER_PATH)

def load_all():
    """Convenience function to load all components."""
    model = load_model()
    preprocessor = load_preprocessor()
    explainer = load_explainer()
    numerical_features = ['Age', 'Annual Income', 'Number of Dependents', 'Health Score',
       'Previous Claims', 'Vehicle Age', 'Credit Score', 'Insurance Duration',
       'Year_sin', 'Year_cos', 'Month_sin', 'Month_cos', 'Day_sin', 'Day_cos']

    categorical_features = ['Gender', 'Marital Status', 'Education Level', 'Occupation', 'Location',
       'Policy Type', 'Customer Feedback', 'Smoking Status',
       'Exercise Frequency', 'Property Type', 'Month_name', 'Day_of_week']

    return model, preprocessor, explainer, numerical_features, categorical_features


def date(df):
    """creates additional features from the date column

    Args:
        df (Dataframe): input dataframe containing the date column

    Returns:
       Dataframe: dataframe with added features
    """
    df['Policy Start Date'] = pd.to_datetime(df['Policy Start Date'])
    df['Year'] = df['Policy Start Date'].dt.year
    df['Day'] = df['Policy Start Date'].dt.day
    df['Month'] = df['Policy Start Date'].dt.month
    df['Month_name'] = df['Policy Start Date'].dt.month_name()
    df['Day_of_week'] = df['Policy Start Date'].dt.day_name()
    df['Week'] = df['Policy Start Date'].dt.isocalendar().week
    df['Year_sin'] = np.sin(2 * np.pi * df['Year'])
    df['Year_cos'] = np.cos(2 * np.pi * df['Year'])
    min_year = df['Year'].min()
    max_year = df['Year'].max()
    df['Year_sin'] = np.sin(2 * np.pi * (df['Year'] - min_year) / (max_year - min_year))
    df['Year_cos'] = np.cos(2 * np.pi * (df['Year'] - min_year) / (max_year - min_year))
    df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12) 
    df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
    df['Day_sin'] = np.sin(2 * np.pi * df['Day'] / 31)  
    df['Day_cos'] = np.cos(2 * np.pi * df['Day'] / 31)
    df['Group']=(df['Year']-2020)*48+df['Month']*4+df['Day']//7
    
    df.drop('Policy Start Date', axis=1, inplace=True)

    return df

def get_feature_names(preprocessor, numerical_features, categorical_features):
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)
    return list(numerical_features) + list(cat_feature_names)

# Prediction logic
def make_prediction(input_df, preprocessor, model):
    input_df = input_df[preprocessor.feature_names_in_]  # Ensure column order
    X_transformed = preprocessor.transform(input_df)
    prediction = model.predict(X_transformed)[0]
    return prediction, X_transformed