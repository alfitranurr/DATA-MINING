Customer Churn Prediction
This project provides a Streamlit-based dashboard to upload a customer churn dataset, explore its features, and predict churn using a pre-trained Random Forest model. Predictions are labeled as "Churn" (1) or "Not Churn" (0).
Folder Structure
customer_churn_prediction/
│
├── app/
│ ├── main.py # Streamlit dashboard script
│ └── requirements.txt # Dependencies for the Streamlit app
│
├── data/
│ └── preprocessed_data.csv # Placeholder for uploaded dataset
│
├── models/
│ └── Random_Forest_model.pkl # Pre-trained Random Forest model
│
└── README.md # Project documentation

Setup Instructions

Clone the Repository:
git clone <repository-url>
cd customer_churn_prediction

Create a Virtual Environment:
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate

Install Dependencies:
pip install -r app/requirements.txt

Place Model File:

Copy Random_Forest_model.pkl to the models/ folder.

Run the Streamlit App:
cd app
streamlit run main.py

Access the Dashboard:Open your browser and navigate to http://localhost:8501.

Dashboard Features

Data Overview: View the shape, sample rows, and basic statistics of the uploaded dataset.
Feature Analysis: Visualize distributions and relationships with churn for selected features.
Churn Analysis: Upload preprocessed_data.csv, click the "Analyze Churn" button to predict churn labels ("Churn" or "Not Churn"), view results in a table, and download predictions as a CSV.

Requirements
See app/requirements.txt for the list of Python packages required.
Notes

Ensure Random_Forest_model.pkl is in the models/ folder before running the app.
Upload preprocessed_data.csv via the dashboard to enable data exploration and analysis.
The dashboard assumes the model is trained on the same features as in the uploaded dataset (excluding the 'Churn' column for predictions).
