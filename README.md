# Customer NPS Prediction for Telecom Operator

## Overview
This application predicts a customer's Net Promoter Score (NPS) category—Detractor, Passive, or Promoter—based on their profile and service usage data. Built for Telecom operators, this tool helps identify potential detractors early for proactive customer service outreach.

## Features
- **Predictive Modeling**: Utilizes an XGBoost machine learning model to classify customers.
- **Interactive Interface**: A clean, intuitive Streamlit web app allowing users to input customer parameters (tenure, charges, contract type, internet type, etc.) in real-time.
- **Probability Scoring**: Displays the confidence percentage for each class, giving deeper insight beyond the final prediction.

## Technology Stack
- **Python**
- **Streamlit** (Web Interface)
- **Scikit-Learn & XGBoost** (Model Training & Prediction)
- **Joblib** (Model Serialization)

## Installation & Usage
1. Clone the repository
2. Install the requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
