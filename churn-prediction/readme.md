# Telco Customer Churn Prediction

This Jupyter notebook outlines a more thorough, step-by-step approach for building a customer churn prediction system using **Kaggle's Telco Customer Churn dataset**. The pipeline includes:

1. **Data Exploration and Cleaning**
2. **Feature Engineering** (including at least 12 features)
3. **Model Training and Baseline Comparison**
4. **Hyperparameter Tuning**
5. **Ensemble Modeling** (Random Forest + XGBoost)
6. **Model Interpretability with SHAP**
7. **Production Deployment** (FastAPI for serving + Streamlit for a simple dashboard)

**Usage Instructions**:

1. **Data Preparation**: Load `"Telco-Customer-Churn.csv"` from kaggle.
2. **Model Saving**: Use `joblib.dump(best_xgb, "best_xgb_model.pkl")` to save the best model.
3. **API**: run `uvicorn api:app --reload`.
4. **Streamlit**: run `streamlit run app.py`.
