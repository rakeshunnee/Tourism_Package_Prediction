import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="rakeshunnee/tourism_package_prediction_model", filename="best_tourism_package_prediction_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for Tourism Package Prediction
st.title("Tourism Package Prediction App")
st.write("""
This application predicts the likelihood of a customer purchasing a tourism package based on their profile.
Please enter the customer data below to get a prediction.
""")

# User input
Age = st.number_input("Age", min_value=1, max_value=100, value=50, step=1)
TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
CityTier = st.number_input("City Tier", min_value=1, max_value=3, value=1, step=1)
DurationOfPitch = st.number_input("Duration Of Pitch", min_value=0.0, max_value=400.0, value=30.0, step=1.0)
Occupation = st.selectbox("Occupation", ["Free Lancer", "Large Business", "Salaried", "Small Business"])
Gender = st.selectbox("Gender", ["Female", "Male"])
NumberOfPersonVisiting = st.number_input("Number Of Person Visiting", min_value=1, max_value=20, value=5, step=1)
NumberOfFollowups = st.number_input("Number Of Followups", min_value=1, max_value=20, value=4, step=1)
ProductPitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "King", "Standard", "Super Deluxe"])
PreferredPropertyStar = st.number_input("Preferred Property Star", min_value=1, max_value=5, value=3, step=1)
MaritalStatus = st.selectbox("Marital Status", ["Married", "Single", "Divorced", "Unmarried"])
NumberOfTrips = st.number_input("Number Of Trips", min_value=1, max_value=50, value=3, step=1)
Passport = st.radio("Do you have a valid passport?", ["Yes", "No"])
PitchSatisfactionScore = st.number_input("Pitch Satisfaction Score", min_value=1.0, max_value=10.0, value=2.0, step=1.0)
own_car = st.radio("Do you own a car?", ["Yes", "No"])
NumberOfChildrenVisiting = st.number_input("Number Of Children Visiting", min_value=0, max_value=10, value=2, step=1)
Designation = st.selectbox("Designation", ["AVP", "Executive", "Manager", "VP", "Senior Manager"])
MonthlyIncome = st.number_input("Monthly Income", min_value=500.0, max_value=200000.0, value=15000.0, step=100.0)

# Convert to model format
Passport = 1 if Passport == "Yes" else 0
own_car = 1 if own_car == "Yes" else 0

# Assemble input into DataFrame
input_data = pd.DataFrame([
{
    'Age': Age,
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'ProductPitched': ProductPitched,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': Passport,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': own_car,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'MonthlyIncome': MonthlyIncome
}])

if st.button("Predict Purchase"): # Changed button text to be more relevant to tourism package
    prediction = model.predict(input_data)[0]
    result = "Purchase Likely" if prediction == 1 else "No Purchase Likely"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
