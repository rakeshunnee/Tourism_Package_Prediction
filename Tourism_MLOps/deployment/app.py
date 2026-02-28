import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="rakeshunnee/tourism_package_prediction_model", filename="best_tourism_package_prediction_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Tourism Package Prediction App")
st.write("""
This application predicts the likelihood of a customer purchasing a tourism package based on their profile.
Please enter the customer data below to get a prediction.
""")

numeric_features = [
    'Age',
    'DurationOfPitch',
    'NumberOfPersonVisiting',
    'NumberOfFollowups',
    'NumberOfTrips',
    'PitchSatisfactionScore',
    'NumberOfChildrenVisiting',
    'MonthlyIncome'
]
categorical_features = [
    'TypeofContact',
    'Occupation',
    'Gender',
    'ProductPitched',
    'MaritalStatus',
    'Designation'
]

# User input
Age = st.number_input("Age", min_value=0, max_value=100.0, value=50.0, step=1)
DurationOfPitch = st.number_input("Duration Of Pitch", min_value=0, max_value=400.0, value=30.0, step=1)
NumberOfPersonVisiting = st.number_input("Number Of Person Visiting", min_value=1.0, max_value=20.0, value=5.0, step=1)
NumberOfFollowups = st.number_input("Number Of Followups", min_value=1.0, max_value=20.0, value=4.0, step=1)
NumberOfTrips = st.number_input("Number Of Trips", min_value=1.0, max_value=50.0, value=3.0, step=1)
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score", min_value=1.0, max_value=10.0, value=2.0, step=1)
NumberOfChildrenVisiting = st.number_input("Number Of Children Visiting", min_value=1.0, max_value=10.0, value=2.0, step=1)
MonthlyIncome = st.number_input("Monthly Income", min_value=500.0, max_value=200000.0, value=15000.0, step=1)

TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
Occupation = st.selectbox("Occupation", ["Free Lancer", "Large Business", "Salaried", "Small Business"])
Gender = st.selectbox("Gender", ["Female", "Male"])
ProductPitched = st.selectbox("Product Pitched", ["Deluxe", "Standard", "Super Deluxe"])
MaritalStatus = st.selectbox("Marital Status", ["Married", "Single"])
Designation = st.selectbox("Designation", ["Executive", "Managerial", "Professional", "Senior Manager"])

# Assemble input into DataFrame
input_data = pd.DataFrame([
{
    'Age': Age,
    'DurationOfPitch': DurationOfPitch,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'NumberOfTrips': NumberOfTrips,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'MonthlyIncome': MonthlyIncome,
    'TypeofContact': TypeofContact,
    'Occupation': Occupation,
    'Gender': Gender,
    'ProductPitched': ProductPitched,
    'MaritalStatus': MaritalStatus,
    'Designation': Designation
}])

if st.button("Predict Purchase"): # Changed button text to be more relevant to tourism package
    prediction = model.predict(input_data)[0]
    result = "Purchase Likely" if prediction == 1 else "No Purchase Likely"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
