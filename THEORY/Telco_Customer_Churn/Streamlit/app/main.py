import streamlit as st
import pandas as pd
import pickle
import os
import math
from sklearn.preprocessing import LabelEncoder
import numpy as np
import plotly.express as px

# Configure Streamlit page
st.set_page_config(page_title="Customer Churn Prediction Dashboard", layout="wide")

# Paths to model and scaler
MODEL_PATH = "../models/Decision_Tree_NonFeatureSelection_model.pkl"
SCALER_PATH = "../models/DT_NonFeatureSelection_scaler.pkl"
CSV_STORAGE_PATH = "manual_input_data.csv"

# Expected feature columns
EXPECTED_FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure", "PhoneService", 
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup", 
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies", 
    "Contract", "PaperlessBilling", "PaymentMethod", "MonthlyCharges", 
    "TotalCharges", "tenure_group"
]

# Function to derive tenure_group from tenure
def derive_tenure_group(tenure):
    if tenure <= 12:
        return "0-12 months"
    elif tenure <= 24:
        return "13-24 months"
    elif tenure <= 36:
        return "25-36 months"
    elif tenure <= 48:
        return "37-48 months"
    elif tenure <= 60:
        return "49-60 months"
    else:
        return "61-72 months"

# Function to load scaler
@st.cache_resource
def load_scaler():
    if not os.path.exists(SCALER_PATH):
        st.error(f"Scaler file not found at {SCALER_PATH}! Ensure the file exists in the models/ folder.")
        return None
    try:
        with open(SCALER_PATH, 'rb') as file:
            scaler = pickle.load(file)
        if not hasattr(scaler, 'transform'):
            st.error(f"Loaded scaler from {SCALER_PATH} is not a valid scaler object (type: {type(scaler)}). "
                     "It lacks a 'transform' method. Recreate and save a proper scaler (e.g., StandardScaler).")
            return None
        return scaler
    except Exception as e:
        st.error(f"Failed to load scaler from {SCALER_PATH}: {str(e)}. Ensure the file contains a valid scaler object.")
        return None

# Function to load model
@st.cache_resource
def load_nb_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at {MODEL_PATH}! Ensure the file exists in the models/ folder.")
        return None
    try:
        with open(MODEL_PATH, 'rb') as file:
            model = pickle.load(file)
        if not hasattr(model, 'predict'):
            st.error(f"Loaded model from {MODEL_PATH} is not a valid model object (type: {type(model)}). "
                     "It lacks a 'predict' method. Recreate and save a proper Naive Bayes model.")
            return None
        return model
    except Exception as e:
        st.error(f"Failed to load model from {MODEL_PATH}: {str(e)}. Ensure the model file is valid.")
        return None

# Load scaler and model
scaler = load_scaler()
model = load_nb_model()

# Function to load existing manual data from CSV
def load_manual_data():
    try:
        if os.path.exists(CSV_STORAGE_PATH):
            return pd.read_csv(CSV_STORAGE_PATH)
        else:
            return pd.DataFrame(columns=EXPECTED_FEATURES)
    except Exception as e:
        st.error(f"Failed to load manual data from CSV: {str(e)}")
        return pd.DataFrame(columns=EXPECTED_FEATURES)

# Function to save manual data to CSV
def save_manual_data(df):
    try:
        df.to_csv(CSV_STORAGE_PATH, index=False)
        st.success("Manual data saved to CSV successfully! ✅")
    except Exception as e:
        st.error(f"Failed to save manual data to CSV: {str(e)}")

# Function to reset manual data
def reset_manual_data():
    try:
        st.session_state['manual_data'] = pd.DataFrame(columns=EXPECTED_FEATURES)
        if os.path.exists(CSV_STORAGE_PATH):
            os.remove(CSV_STORAGE_PATH)
        st.session_state['data'] = None
        st.success("Manual data and CSV storage reset successfully! ♻️")
    except Exception as e:
        st.error(f"Failed to reset manual data: {str(e)}")

# Function to preprocess features
def preprocess_features(df):
    df = df.copy()
    for col in EXPECTED_FEATURES:
        if col not in df.columns:
            if col == 'tenure_group' and 'tenure' in df.columns:
                df['tenure_group'] = df['tenure'].apply(derive_tenure_group)
            else:
                df[col] = 'Male' if col == 'gender' else \
                          'No' if col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 
                                          'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                                          'TechSupport', 'StreamingTV', 'StreamingMovies', 'MultipleLines'] else \
                          'Fiber optic' if col == 'InternetService' else \
                          '0-12 months' if col == 'tenure_group' else \
                          'Month-to-month' if col == 'Contract' else \
                          'Electronic check' if col == 'PaymentMethod' else 0
    
    label_encoders = {}
    categorical_columns = [
        "gender", "Partner", "Dependents", "PhoneService", "MultipleLines", 
        "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection", 
        "TechSupport", "StreamingTV", "StreamingMovies", "Contract", 
        "PaperlessBilling", "PaymentMethod", "tenure_group"
    ]
    
    for col in categorical_columns:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    
    numerical_columns = ["SeniorCitizen", "tenure", "MonthlyChargesทุ", "TotalCharges"]
    for col in numerical_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    df = df[EXPECTED_FEATURES]
    return df, label_encoders

# Initialize session state
if 'data' not in st.session_state:
    st.session_state['data'] = None
if 'input_method' not in st.session_state:
    st.session_state['input_method'] = None
if 'manual_data' not in st.session_state:
    st.session_state['manual_data'] = load_manual_data()

# Sidebar for navigation
st.sidebar.header("Navigation 📋")
page = st.sidebar.radio("Select Page", ["Home", "Data Overview", "Churn Analysis"])

# Home Page
if page == "Home":
    st.title("Customer Churn Prediction Dashboard 🏠")
    st.markdown("""
    **Welcome to the Customer Churn Prediction Dashboard**  
    Unlock actionable insights with this advanced, user-friendly platform designed for telecom businesses. 
    Seamlessly upload customer datasets (CSV or XLSX) or manually input data to explore key features, 
    analyze patterns, and predict churn outcomes ("Churn" or "Not Churn") with precision. 
    Powered by a robust Decision Tree model, this professional-grade dashboard empowers you to make 
    data-driven decisions to enhance customer retention and drive strategic growth.
    """)
    
    st.header("Input Method 📥")
    input_method = st.radio("Choose Input Method", ["Upload File", "Manual Input"])
    st.session_state['input_method'] = input_method

    if input_method == "Upload File":
        st.header("Upload Dataset 📂")
        uploaded_file = st.file_uploader("Upload data as CSV or XLSX", type=["csv", "xlsx"])
        
        @st.cache_data
        def load_data(file):
            try:
                if file.name.endswith('.csv'):
                    df = pd.read_csv(file)
                elif file.name.endswith('.xlsx'):
                    try:
                        import openpyxl
                    except ImportError:
                        st.error("Missing optional dependency 'openpyxl'. Install using 'pip install openpyxl'.")
                        return None
                    df = pd.read_excel(file)
                else:
                    st.error("Unsupported file format. Please upload a CSV or XLSX file.")
                    return None
                return df
            except Exception as e:
                st.error(f"Failed to load dataset: {str(e)}.")
                return None
        
        if uploaded_file is not None:
            data = load_data(uploaded_file)
            if data is not None:
                for col in EXPECTED_FEATURES:
                    if col not in data.columns:
                        if col == 'tenure_group' and 'tenure' in data.columns:
                            data['tenure_group'] = data['tenure'].apply(derive_tenure_group)
                        else:
                            data[col] = 'Male' if col == 'gender' else \
                                        'No' if col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 
                                                        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                                                        'TechSupport', 'StreamingTV', 'StreamingMovies', 'MultipleLines'] else \
                                        'Fiber optic' if col == 'InternetService' else \
                                        '0-12 months' if col == 'tenure_group' else \
                                        'Month-to-month' if col == 'Contract' else \
                                        'Electronic check' if col == 'PaymentMethod' else 0
                st.session_state['data'] = data
                st.success("Dataset loaded and additional columns verified/added. ✅")
            else:
                st.session_state['data'] = None
                st.error("Failed to load the uploaded dataset. Check the file format.")

    elif input_method == "Manual Input":
        st.header("Manual Customer Data Input ✍️")
        st.markdown("Enter values for each feature below.")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            tenure = st.number_input("Tenure (months)", min_value=0, value=0)
        
        with col2:
            phone_service = st.selectbox("Phone Service", ["No", "Yes"])
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
        
        with col3:
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        
        with col4:
            paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"])
            payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", 
                                                             "Bank transfer (automatic)", "Credit card (automatic)"])
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=0.0, step=0.1)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=0.0, step=0.1)
            tenure_group = st.selectbox("Tenure Group", ["0-12 months", "13-24 months", "25-36 months", 
                                                         "37-48 months", "49-60 months", "61-72 months"])
        
        features = {
            "gender": gender,
            "SeniorCitizen": senior_citizen,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet_service,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless_billing,
            "PaymentMethod": payment_method,
            "MonthlyCharges": monthly_charges,
            "TotalCharges": total_charges,
            "tenure_group": tenure_group
        }
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        with col_btn1:
            if st.button("Add Data"):
                try:
                    new_entry = pd.DataFrame([features])
                    st.session_state['manual_data'] = pd.concat([st.session_state['manual_data'], new_entry], ignore_index=True)
                    st.session_state['data'] = st.session_state['manual_data']
                    st.success("Data added successfully! ✅")
                except Exception as e:
                    st.error(f"Failed to add data: {str(e)}")
        
        with col_btn2:
            if st.button("Submit Data"):
                try:
                    save_manual_data(st.session_state['manual_data'])
                    st.session_state['data'] = st.session_state['manual_data']
                    st.success("Data submitted and saved to CSV successfully! ✅")
                except Exception as e:
                    st.error(f"Failed to submit data to CSV: {str(e)}")
        
        with col_btn3:
            if st.button("Reset Data"):
                reset_manual_data()
        
        st.subheader("Current Manual Data 📊")
        if not st.session_state['manual_data'].empty:
            rows_per_page = st.session_state.get("home_rows_per_page", 10)
            total_rows = len(st.session_state['manual_data'])
            total_pages = math.ceil(total_rows / rows_per_page) if total_rows > 0 else 1
            current_page = st.session_state.get("home_page", 1)
            start_idx = (current_page - 1) * rows_per_page
            end_idx = min(start_idx + rows_per_page, total_rows)
            
            display_data = st.session_state['manual_data'].iloc[start_idx:end_idx].copy()
            
            st.dataframe(display_data.style.set_properties(**{'min-width': '100px', 'text-align': 'left'}),
                         height=300, use_container_width=True)
            
            col1, col2 = st.columns([1, 1])
            with col1:
                rows_per_page = st.selectbox("Rows per page", [5, 10, 25, 50, 100], 
                                             index=[5, 10, 25, 50, 100].index(rows_per_page), 
                                             key="home_rows_per_page")
            with col2:
                current_page = st.selectbox("Page", list(range(1, total_pages + 1)), 
                                            index=current_page - 1, 
                                            key="home_page")
        
        if 'edit_row' in st.session_state:
            st.subheader(f"Edit Row ✏️")
            edit_data = st.session_state['edit_data']
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                gender = st.selectbox("Gender", ["Male", "Female"], 
                                      index=["Male", "Female"].index(edit_data.get("gender", "Male")), 
                                      key="edit_gender")
                senior_citizen = st.selectbox("Senior Citizen", [0, 1], 
                                             index=[0, 1].index(int(edit_data.get("SeniorCitizen", 0))), 
                                             key="edit_senior")
                partner = st.selectbox("Partner", ["No", "Yes"], 
                                      index=["No", "Yes"].index(edit_data.get("Partner", "No")), 
                                      key="edit_partner")
                dependents = st.selectbox("Dependents", ["No", "Yes"], 
                                         index=["No", "Yes"].index(edit_data.get("Dependents", "No")), 
                                         key="edit_dependents")
                tenure = st.number_input("Tenure (months)", min_value=0, value=int(edit_data.get("tenure", 0)), 
                                        key="edit_tenure")
            with col2:
                phone_service = st.selectbox("Phone Service", ["No", "Yes"], 
                                            index=["No", "Yes"].index(edit_data.get("PhoneService", "No")), 
                                            key="edit_phone")
                multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"], 
                                             index=["No", "Yes", "No phone service"].index(edit_data.get("MultipleLines", "No")), 
                                             key="edit_multiple")
                internet_service = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"], 
                                               index=["Fiber optic", "DSL", "No"].index(edit_data.get("InternetService", "Fiber optic")), 
                                               key="edit_internet")
                online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"], 
                                              index=["No", "Yes", "No internet service"].index(edit_data.get("OnlineSecurity", "No")), 
                                              key="edit_security")
                online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"], 
                                            index=["No", "Yes", "No internet service"].index(edit_data.get("OnlineBackup", "No")), 
                                            key="edit_backup")
            with col3:
                device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"], 
                                                index=["No", "Yes", "No internet service"].index(edit_data.get("DeviceProtection", "No")), 
                                                key="edit_protection")
                tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"], 
                                           index=["No", "Yes", "No internet service"].index(edit_data.get("TechSupport", "No")), 
                                           key="edit_support")
                streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"], 
                                           index=["No", "Yes", "No internet service"].index(edit_data.get("StreamingTV", "No")), 
                                           key="edit_streaming_tv")
                streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"], 
                                               index=["No", "Yes", "No internet service"].index(edit_data.get("StreamingMovies", "No")), 
                                               key="edit_streaming_movies")
                contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"], 
                                       index=["Month-to-month", "One year", "Two year"].index(edit_data.get("Contract", "Month-to-month")), 
                                       key="edit_contract")
            with col4:
                paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"], 
                                                index=["No", "Yes"].index(edit_data.get("PaperlessBilling", "No")), 
                                                key="edit_paperless")
                payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", 
                                                                "Bank transfer (automatic)", "Credit card (automatic)"], 
                                             index=["Electronic check", "Mailed check", "Bank transfer (automatic)", 
                                                    "Credit card (automatic)"].index(edit_data.get("PaymentMethod", "Electronic check")), 
                                             key="edit_payment")
                monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=float(edit_data.get("MonthlyCharges", 0.0)), 
                                                 step=0.1, key="edit_monthly")
                total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=float(edit_data.get("TotalCharges", 0.0)), 
                                               step=0.1, key="edit_total")
                tenure_group = st.selectbox("Tenure Group", ["0-12 months", "13-24 months", "25-36 months", 
                                                             "37-48 months", "49-60 months", "61-72 months"], 
                                            index=["0-12 months", "13-24 months", "25-36 months", 
                                                   "37-48 months", "49-60 months", "61-72 months"].index(edit_data.get("tenure_group", "0-12 months")), 
                                            key="edit_tenure_group")
            
            edited_features = {
                "gender": gender,
                "SeniorCitizen": senior_citizen,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless_billing,
                "PaymentMethod": payment_method,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
                "tenure_group": tenure_group
            }
            if st.button("Save Changes"):
                try:
                    st.session_state['manual_data'].loc[st.session_state['edit_row']] = pd.Series(edited_features)
                    save_manual_data(st.session_state['manual_data'])
                    st.session_state['data'] = st.session_state['manual_data']
                    st.success(f"Row updated successfully! ✅")
                    del st.session_state['edit_row']
                    del st.session_state['edit_data']
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to update row: {str(e)}")

# Custom CSS for button colors and horizontal scrolling
st.markdown("""
<style>
button[kind="primary"][data-testid="baseButton-primary"]:has(span:contains(":wastebasket:")) {
    background-color: #ff0000;
    color: white;
}
button[kind="primary"][data-testid="baseButton-primary"]:not(:has(span:contains(":wastebasket:"))) {
    background-color: #ffd700;
    color: black;
}
div[data-testid="stDataFrame"] {
    overflow-x: auto;
    max-width: 100%;
}
div[data-testid="stDataFrame"] table {
    width: max-content;
}
</style>
""", unsafe_allow_html=True)

# Check if data is available
if st.session_state['data'] is not None and not st.session_state['data'].empty:
    if page == "Data Overview":
        st.header("Data Overview 📈")
        st.write("Dataset Size:", st.session_state['data'].shape)
        
        st.subheader("Current Dataset 📋")
        rows_per_page = st.session_state.get("overview_rows_per_page", 10)
        total_rows = len(st.session_state['data'])
        total_pages = math.ceil(total_rows / rows_per_page) if total_rows > 0 else 1
        current_page = st.session_state.get("overview_page", 1)
        start_idx = (current_page - 1) * rows_per_page
        end_idx = min(start_idx + rows_per_page, total_rows)
        
        display_data = st.session_state['data'].iloc[start_idx:end_idx].copy()
        
        st.dataframe(display_data.style.set_properties(**{'min-width': '100px', 'text-align': 'left'}),
                     height=300, use_container_width=True)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            rows_per_page = st.selectbox("Rows per page", [5, 10, 25, 50, 100], 
                                         index=[5, 10, 25, 50, 100].index(rows_per_page), 
                                         key="overview_rows_per_page")
        with col2:
            current_page = st.selectbox("Page", list(range(1, total_pages + 1)), 
                                        index=current_page - 1, 
                                        key="overview_page")
        
        st.subheader("Basic Statistics 📊")
        st.write("Statistics for current data:")
        st.dataframe(st.session_state['data'].describe() if st.session_state['data'].shape[0] > 1 else st.session_state['data'])
        
        csv = st.session_state['data'].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Current Dataset 📥",
            data=csv,
            file_name="current_data.csv",
            mime="text/csv"
        )

    elif page == "Churn Analysis":
        st.header("Churn Analysis 🔍")
        if model is not None and scaler is not None:
            if st.button("Analyze Churn"):
                features = st.session_state['data'].drop(['Churn', 'Predicted_Churn'], axis=1, errors='ignore')
                
                try:
                    features_processed, _ = preprocess_features(features)
                except Exception as e:
                    st.error(f"Failed to preprocess features: {str(e)}.")
                    st.stop()
                
                if hasattr(scaler, 'feature_names_in_'):
                    scaler_features = set(scaler.feature_names_in_)
                    input_features = set(features_processed.columns)
                    missing_features = scaler_features - input_features
                    extra_features = input_features - scaler_features
                    if missing_features or extra_features:
                        error_msg = ""
                        if missing_features:
                            error_msg += f"Missing features: {missing_features}\n"
                        if extra_features:
                            error_msg += f"Unexpected features: {extra_features}"
                        st.error(f"Feature mismatch:\n{error_msg}")
                        st.stop()
                
                try:
                    features_scaled = scaler.transform(features_processed)
                except Exception as e:
                    st.error(f"Failed to scale features: {str(e)}. Ensure the scaler is valid and compatible with the data.")
                    st.stop()

                try:
                    predictions = model.predict(features_scaled)
                    # Check if model supports predict_proba
                    if hasattr(model, 'predict_proba'):
                        probabilities = model.predict_proba(features_scaled)
                    else:
                        probabilities = np.zeros((len(predictions), 2))  # Dummy array if predict_proba is unavailable
                except Exception as e:
                    st.error(f"Failed to make predictions: {str(e)}. Ensure the model is valid and compatible with the scaled data.")
                    st.stop()

                label_map = {0: "Not Churn", 1: "Churn"}
                predicted_labels = [label_map[pred] for pred in predictions]
                
                result_df = st.session_state['data'].copy()
                result_df['Predicted_Churn'] = predicted_labels
                if hasattr(model, 'predict_proba'):
                    result_df['Churn_Probability'] = probabilities[:, 1]  # Probability of Churn (class 1)
                
                st.subheader("Prediction Results 📋")
                st.write("The table below shows the dataset with predicted churn labels:")
                
                # Pagination for Prediction Results
                rows_per_page = st.session_state.get("churn_rows_per_page", 10)
                total_rows = len(result_df)
                total_pages = math.ceil(total_rows / rows_per_page) if total_rows > 0 else 1
                current_page = st.session_state.get("churn_page", 1)
                start_idx = (current_page - 1) * rows_per_page
                end_idx = min(start_idx + rows_per_page, total_rows)
                
                display_data = result_df.iloc[start_idx:end_idx].copy()
                
                st.dataframe(display_data.style.set_properties(**{'min-width': '100px', 'text-align': 'left'}),
                             height=300, use_container_width=True)
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    rows_per_page = st.selectbox("Rows per page", [5, 10, 25, 50, 100], 
                                                 index=[5, 10, 25, 50, 100].index(rows_per_page), 
                                                 key="churn_rows_per_page")
                with col2:
                    current_page = st.selectbox("Page", list(range(1, total_pages + 1)), 
                                                index=current_page - 1, 
                                                key="churn_page")
                
                st.subheader("Prediction Summary 📊")
                if st.session_state.get('input_method') == "Manual Input":
                    st.write("Predicted Churn for Manual Input:")
                    for i, label in enumerate(predicted_labels):
                        prob = probabilities[i, 1] if hasattr(model, 'predict_proba') else "N/A"
                        st.write(f"Instance {i+1}: {label} (Churn Probability: {prob:.2%} if available)")
                else:
                    churn_counts = pd.Series(predicted_labels).value_counts()
                    st.write("Number of Customers Predicted as Not Churn:", churn_counts.get("Not Churn", 0))
                    st.write("Number of Customers Predicted as Churn:", churn_counts.get("Churn", 0))
                
                # Bar Chart for Churn and Not Churn Percentages
                st.subheader("Churn Prediction Distribution 📊")
                churn_counts = pd.Series(predicted_labels).value_counts(normalize=True) * 100
                chart_data = pd.DataFrame({
                    "Category": churn_counts.index,
                    "Percentage": churn_counts.values
                })
                fig = px.bar(chart_data, x="Category", y="Percentage", 
                             title="Percentage of Churn vs Not Churn",
                             labels={"Percentage": "Percentage (%)"},
                             color="Category",
                             color_discrete_map={"Churn": "#FF6384", "Not Churn": "#36A2EB"},
                             text="Percentage")
                fig.update_traces(
                    texttemplate='<b>%{text:.1f}%</b>',
                    textposition='auto',
                    textfont=dict(size=18, color="white"),
                    width=0.5
                )
                fig.update_layout(
                    yaxis_title="Percentage (%)",
                    xaxis_title="Prediction",
                    showlegend=True,
                    bargap=0.4,
                    height=400,
                    font=dict(size=14),
                    title_x=0.5,
                    margin=dict(l=50, r=50, t=80, b=50)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Classification Accuracy (if actual 'Churn' labels are available)
                st.subheader("Classification Accuracy 📏")
                if 'Churn' in st.session_state['data'].columns:
                    actual_labels = st.session_state['data']['Churn'].astype(str).str.strip()
                    # Map actual labels to match predicted format if necessary
                    actual_labels = actual_labels.replace({'Yes': 'Churn', 'No': 'Not Churn'})
                    correct_predictions = sum(actual_labels == pd.Series(predicted_labels))
                    total_predictions = len(predicted_labels)
                    accuracy = (correct_predictions / total_predictions) * 100
                    error_rate = 100 - accuracy
                    
                    st.write(f"Correct Predictions: {correct_predictions} out of {total_predictions}")
                    st.write(f"Accuracy (Correct Classifications): {accuracy:.2f}% ✅")
                    st.write(f"Error Rate (Incorrect Classifications): {error_rate:.2f}% ❌")
                    
                    # Bar Chart for Accuracy and Error Rate
                    accuracy_data = pd.DataFrame({
                        "Category": ["Correct", "Incorrect"],
                        "Percentage": [accuracy, error_rate]
                    })
                    fig_accuracy = px.bar(accuracy_data, x="Category", y="Percentage", 
                                         title="Classification Accuracy vs Error Rate",
                                         labels={"Percentage": "Percentage (%)"},
                                         color="Category",
                                         color_discrete_map={"Correct": "#36A2EB", "Incorrect": "#FF6384"},
                                         text="Percentage")
                    fig_accuracy.update_traces(
                        texttemplate='<b>%{text:.1f}%</b>',
                        textposition='auto',
                        textfont=dict(size=18, color="white"),
                        width=0.5
                    )
                    fig_accuracy.update_layout(
                        yaxis_title="Percentage (%)",
                        xaxis_title="Classification Outcome",
                        showlegend=True,
                        bargap=0.4,
                        height=400,
                        font=dict(size=14),
                        title_x=0.5,
                        margin=dict(l=50, r=50, t=80, b=50)
                    )
                    st.plotly_chart(fig_accuracy, use_container_width=True)
                else:
                    st.warning("No 'Churn' column found in the dataset. Accuracy metrics require actual churn labels to compare against predictions. ⚠️")
                
                # Churn Probability Distribution
                if hasattr(model, 'predict_proba'):
                    st.subheader("Churn Probability Distribution 📈")
                    prob_data = pd.DataFrame({
                        "Instance": [f"Instance {i+1}" for i in range(len(probabilities))],
                        "Churn_Probability": probabilities[:, 1] * 100
                    })
                    fig_prob = px.histogram(prob_data, x="Churn_Probability", 
                                           title="Distribution of Churn Probabilities",
                                           labels={"Churn_Probability": "Churn Probability (%)"},
                                           nbins=20, color_discrete_sequence=["#FF6384"])
                    fig_prob.update_layout(
                        yaxis_title="Count",
                        xaxis_title="Churn Probability (%)",
                        height=400,
                        font=dict(size=14),
                        title_x=0.5,
                        margin=dict(l=50, r=50, t=80, b=50)
                    )
                    st.plotly_chart(fig_prob, use_container_width=True)
                else:
                    st.warning("Model does not support probability predictions (no predict_proba method). ⚠️")
                
                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Prediction Results 📥",
                    data=csv,
                    file_name="churn_predictions.csv",
                    mime="text/csv"
                )
                
                if st.session_state.get('input_method') == "Manual Input":
                    st.session_state['manual_data'] = result_df
                    save_manual_data(st.session_state['manual_data'])
                    st.session_state['data'] = st.session_state['manual_data']
        else:
            st.error("Cannot perform analysis. Ensure both the Naive Bayes model and scaler are loaded correctly and valid. ⚠️")
else:
    if page != "Home":
        st.warning("Please upload a CSV or XLSX file or add manual data in the Home page to continue. ⚠️")

if model is None or scaler is None:
    st.error(f"Ensure both {MODEL_PATH} and {SCALER_PATH} exist in the models/ folder and contain valid objects. ⚠️")