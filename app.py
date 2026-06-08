import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(page_title="NPS Predictor", page_icon="📡", layout="centered")

st.title("📡 Telecom NPS Predictor")
st.markdown("""
This tool predicts the Net Promoter Score (NPS) category of a customer 
(Detractor, Passive, or Promoter) based on their profile.
""")

@st.cache_resource
def load_models():
    preprocessor = joblib.load('preprocessor.joblib')
    xgb_model = joblib.load('xgb_model.joblib')
    return preprocessor, xgb_model

preprocessor, xgb_model = load_models()

# Mapping
class_mapping = {0: 'Detractor', 1: 'Passive', 2: 'Promoter'}

st.header("Customer Profile")

col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input("Tenure (Months)", min_value=0, max_value=120, value=12)
    monthly_charge = st.number_input("Monthly Charge ($)", min_value=0.0, max_value=300.0, value=70.0)
    contract = st.selectbox("Contract", ["Month-to-Month", "One Year", "Two Year"])
    internet_type = st.selectbox("Internet Type", ["Fiber Optic", "DSL", "Cable", "None"])
    senior_citizen = st.selectbox("Senior Citizen", ["Yes", "No"])

with col2:
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    num_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=0)
    payment_method = st.selectbox("Payment Method", ["Bank Withdrawal", "Credit Card", "Mailed Check"])
    tech_support = st.selectbox("Premium Tech Support", ["Yes", "No"])
    total_charges = tenure * monthly_charge

if st.button("Predict NPS Category", type="primary"):
    # Create a dummy dataframe with the exact columns needed
    # We will use the original dataset's columns and fill defaults for others
    # just for the demonstration purposes.
    
    # In a real app, you would collect all features or fill median/mode
    # For this challenge, we will create a dictionary with mostly neutral defaults
    # and update with user input
    st.write("Generating Prediction...")
    
    input_data = {
        'Age': 40, 'Number of Dependents': num_dependents, 'Latitude': 34.0, 'Longitude': -118.0,
        'Population': 10000, 'Number of Referrals': 0, 'Tenure in Months': tenure,
        'Avg Monthly Long Distance Charges': 10.0, 'Avg Monthly GB Download': 20.0,
        'Monthly Charge': monthly_charge, 'Total Charges': total_charges,
        'Total Refunds': 0.0, 'Total Extra Data Charges': 0.0,
        'Total Long Distance Charges': 0.0, 'Total Revenue': total_charges,
        'Gender': 'Female', 'Under 30': 'No', 'Senior Citizen': senior_citizen,
        'Married': 'No', 'Dependents': dependents, 'City': 'Los Angeles',
        'Referred a Friend': 'No', 'Offer': 'None', 'Phone Service': 'Yes',
        'Multiple Lines': 'No', 'Internet Service': 'Yes' if internet_type != 'None' else 'No',
        'Internet Type': internet_type, 'Online Security': 'No', 'Online Backup': 'No',
        'Device Protection Plan': 'No', 'Premium Tech Support': tech_support,
        'Streaming TV': 'No', 'Streaming Movies': 'No', 'Streaming Music': 'No',
        'Unlimited Data': 'Yes', 'Contract': contract, 'Paperless Billing': 'Yes',
        'Payment Method': payment_method
    }
    
    df_input = pd.DataFrame([input_data])
    
    # Transform and predict
    try:
        X_trans = preprocessor.transform(df_input)
        pred_num = xgb_model.predict(X_trans)[0]
        pred_label = class_mapping[pred_num]
        
        probs = xgb_model.predict_proba(X_trans)[0]
        
        st.success(f"**Predicted Category:** {pred_label}")
        
        st.write("### Probabilities")
        st.write(f"- Detractor: {probs[0]:.1%}")
        st.write(f"- Passive: {probs[1]:.1%}")
        st.write(f"- Promoter: {probs[2]:.1%}")
        
        if pred_label == 'Detractor':
            st.warning("⚠️ This customer is likely a Detractor. Consider proactive outreach.")
            
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
