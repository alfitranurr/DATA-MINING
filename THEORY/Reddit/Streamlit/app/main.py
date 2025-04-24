import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configure Streamlit page
st.set_page_config(page_title="Customer Churn Prediction Dashboard", layout="wide")

# Title and description
st.title("Customer Churn Prediction Dashboard")
st.markdown("""
This dashboard allows you to upload a customer churn dataset, explore its features,
and predict churn labels ("Churn" or "Not Churn") using a pre-trained ANN model.
Ensure the `ANN_model.h5` and `scaler.pkl` files are in the `models/` folder, and upload the dataset via the sidebar.
""")

# Paths to model and scaler
MODEL_PATH = "../models/ANN_model.h5"
SCALER_PATH = "../models/scaler.pkl"

# Function to load scaler
@st.cache_resource
def load_scaler():
    if not os.path.exists(SCALER_PATH):
        st.error(f"Scaler file not found at {SCALER_PATH}! Ensure the file exists in the models/ folder.")
        return None
    try:
        with open(SCALER_PATH, 'rb') as file:
            scaler = pickle.load(file)
        st.write("Scaler successfully loaded.")
        return scaler
    except Exception as e:
        st.error(f"Failed to load scaler: {str(e)}. Ensure the scaler file is valid.")
        return None

# Function to load model
@st.cache_resource
def load_ann_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}! Ensure the file exists in the models/ folder.")
        return None
    try:
        model = load_model(MODEL_PATH)
        st.write("ANN model successfully loaded.")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}. Ensure the model file is valid.")
        return None

# Load scaler and model
scaler = load_scaler()
model = load_ann_model()

# File uploader for dataset
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload data as CSV", type=["csv"])

# Sidebar for navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select Page", ["Data Overview", "Feature Analysis", "Churn Analysis"])

# Check if file is uploaded
if uploaded_file is not None:
    # Function to load data
    @st.cache_data
    def load_data(file):
        try:
            df = pd.read_csv(file)
            return df
        except Exception as e:
            st.error(f"Failed to load dataset: {str(e)}. Ensure the CSV file is valid.")
            return None

    data = load_data(uploaded_file)

    if data is not None:
        # Data Overview Page
        if page == "Data Overview":
            st.header("Data Overview")
            st.write("Dataset Size:", data.shape)
            st.write("First 5 Rows of Dataset:")
            st.dataframe(data.head())
            
            # Basic statistics
            st.subheader("Basic Statistics")
            st.write(data.describe())

        # Feature Analysis Page
        elif page == "Feature Analysis":
            st.header("Feature Analysis")
            feature = st.selectbox("Select Feature for Visualization", data.columns)
            
            # Plot distribution
            st.subheader(f"Distribution of {feature}")
            fig, ax = plt.subplots()
            if data[feature].dtype in ['int64', 'float64']:
                sns.histplot(data[feature], kde=True, ax=ax)
            else:
                sns.countplot(x=feature, data=data, ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)

            # Relationship with Churn
            if feature != 'Churn' and 'Churn' in data.columns:
                st.subheader(f"Relationship with Churn")
                fig, ax = plt.subplots()
                if data[feature].dtype in ['int64', 'float64']:
                    sns.boxplot(x='Churn', y=feature, data=data, ax=ax)
                else:
                    sns.countplot(x=feature, hue='Churn', data=data, ax=ax)
                plt.xticks(rotation=45)
                st.pyplot(fig)

        # Churn Analysis Page
        elif page == "Churn Analysis":
            st.header("Churn Analysis")
            if model is not None and scaler is not None:
                # Button to trigger analysis
                if st.button("Analyze Churn"):
                    # Prepare features for prediction
                    features = data.drop('Churn', axis=1, errors='ignore')
                    
                    # Scale features
                    try:
                        features_scaled = scaler.transform(features)
                    except Exception as e:
                        st.error(f"Failed to scale features: {str(e)}. Ensure the dataset columns match the training data.")
                        st.stop()

                    # Predict using ANN
                    try:
                        predictions = (model.predict(features_scaled) > 0.5).astype(int).flatten()
                    except Exception as e:
                        st.error(f"Failed to make predictions: {str(e)}. Ensure the dataset features match the model's input.")
                        st.stop()

                    # Map predictions to labels
                    label_map = {0: "Not Churn", 1: "Churn"}
                    predicted_labels = [label_map[pred] for pred in predictions]
                    
                    # Create result DataFrame
                    result_df = data.copy()
                    result_df['Predicted_Churn'] = predicted_labels
                    
                    # Display results
                    st.subheader("Prediction Results")
                    st.write("The table below shows the dataset with predicted churn labels:")
                    st.dataframe(result_df)
                    
                    # Prediction summary
                    st.subheader("Prediction Summary")
                    churn_counts = pd.Series(predicted_labels).value_counts()
                    st.write("Number of Customers Predicted as Not Churn:", churn_counts.get("Not Churn", 0))
                    st.write("Number of Customers Predicted as Churn:", churn_counts.get("Churn", 0))
                    
                    # Download results
                    csv = result_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Prediction Results",
                        data=csv,
                        file_name="churn_predictions.csv",
                        mime="text/csv"
                    )
            else:
                st.error("Cannot perform analysis. Ensure both the ANN model and scaler are loaded correctly.")
    else:
        st.error("Failed to load the uploaded dataset. Check the CSV file format.")
else:
    st.warning("Please upload a preprocessed_data.csv file to continue.")

if model is None or scaler is None:
    st.error(f"Ensure both {MODEL_PATH} and {SCALER_PATH} exist in the models/ folder.")