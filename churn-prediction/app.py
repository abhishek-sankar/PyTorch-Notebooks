import streamlit as st
import numpy as np
import joblib

st.title("Telco Churn Prediction Dashboard")

model = joblib.load("best_xgb_model.pkl")

tenure = st.slider("Tenure (months)", 0, 72, 10)
monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0, step=1.0)
total_charges = st.number_input("Total Charges", 0.0, 10000.0, 500.0, step=50.0)
senior_citizen = st.selectbox("Senior Citizen?", ["No", "Yes"])
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
dependents = st.selectbox("Dependents?", ["No", "Yes"])

if st.button("Predict"):
    sc_binary = 1 if senior_citizen == "Yes" else 0
    contract_one = 1 if contract == "One year" else 0
    contract_two = 1 if contract == "Two year" else 0
    dep_binary = 1 if dependents == "Yes" else 0

    # [tenure, monthly_charges, total_charges, sc_binary, contract_one, contract_two, dep_binary]
    input_features = np.array(
        [
            tenure,
            monthly_charges,
            total_charges,
            sc_binary,
            contract_one,
            contract_two,
            dep_binary,
        ]
    ).reshape(1, -1)
    prediction = model.predict(input_features)
    churn_label = "Yes" if prediction[0] == 1 else "No"
    st.write(f"**Churn Prediction**: {churn_label}")
