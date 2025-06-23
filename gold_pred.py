import streamlit as st
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Load your cleaned dataset
@st.cache_data
def load_data():
    df = pd.read_csv('df_new.csv')  # Replace with your actual CSV file
    df['Close'] = pd.to_numeric(df['Close'].astype(str).str.replace(',', ''), errors='coerce')
    df = df[['Gold_Price', 'Close', 'USD_INR']].dropna()
    return df

df = load_data()

# Train the model
X = df[['Close', 'USD_INR']]
y = df['Gold_Price']
model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
model.fit(X, y)

# Streamlit UI
st.title("💰 Gold Price Predictor")
st.markdown("Predict future gold price based on Sensex and USD-INR inputs.")

# Inputs
sensex = st.number_input("Enter expected Sensex (Close)", min_value=20000, max_value=100000, value=75000)
usd_inr = st.number_input("Enter expected USD-INR exchange rate", min_value=60.0, max_value=100.0, value=83.0)

# Predict
input_df = pd.DataFrame({'Close': [sensex], 'USD_INR': [usd_inr]})
predicted_price = model.predict(input_df)[0]

# Output
st.success(f"📈 Predicted Gold Price (INR): ₹{predicted_price:,.2f}")