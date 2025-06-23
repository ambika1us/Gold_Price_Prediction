import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("df_new.csv")  # Replace with your data path
    df['Close'] = pd.to_numeric(df['Close'].astype(str).str.replace(',', ''), errors='coerce')
    return df[['Gold_Price', 'Close', 'USD_INR']].dropna()

df = load_data()
X = df[['Close', 'USD_INR']]
y = df['Gold_Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tuned Random Forest & XGBoost
rf = RandomForestRegressor(n_estimators=200, max_depth=5, min_samples_split=2, random_state=42)
xgb = XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.1, subsample=1.0, random_state=42)

rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)

# Streamlit UI
st.title("📈 Gold Price Predictor")
st.markdown("Compare predicted gold prices using **Random Forest** and **XGBoost**, and understand model decisions with SHAP.")

sensex = st.number_input("Enter Sensex (Close)", value=85000.0)
usd_inr = st.number_input("Enter USD/INR", value=87.0)

input_df = pd.DataFrame({'Close': [sensex], 'USD_INR': [usd_inr]})
rf_pred = rf.predict(input_df)[0]
xgb_pred = xgb.predict(input_df)[0]

st.success(f"🌳 Random Forest Prediction: ₹{rf_pred:,.2f}")
st.success(f"⚡ XGBoost Prediction: ₹{xgb_pred:,.2f}")

# SHAP explanations
with st.expander("🔍 View SHAP Interpretations"):
    explainer_rf = shap.Explainer(rf, X_train)
    shap_values_rf = explainer_rf(input_df)

    explainer_xgb = shap.Explainer(xgb, X_train)
    shap_values_xgb = explainer_xgb(input_df)

    st.subheader("Random Forest SHAP")
    fig_rf, ax_rf = plt.subplots()
    shap.plots.waterfall(shap_values_rf[0], show=False)
    st.pyplot(fig_rf, bbox_inches='tight', dpi=300, use_container_width=True)

    st.subheader("XGBoost SHAP")
    fig_xgb, ax_xgb = plt.subplots()
    shap.plots.waterfall(shap_values_rf[0], show=False)
    st.pyplot(fig_xgb, bbox_inches='tight', dpi=300, use_container_width=True)

    #fig_xgb = shap.plots.waterfall(shap_values_xgb[0], show=False)
    #st.pyplot(bbox_inches='tight', dpi=300, use_container_width=True)