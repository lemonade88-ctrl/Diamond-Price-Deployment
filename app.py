import streamlit as st
import streamlit.components.v1 as stc
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# Load model dan preprocessing
with open('xgbr_model.pkl', 'rb') as file:
    XGBR_Model = pickle.load(file)

with open('feature_columns.pkl', 'rb') as file:
    feature_columns = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)

# Header HTML
html_temp = """
<div style="background-color:#000;padding:20px;border-radius:10px">
    <h1 style="color:#00ffcc;text-align:center;font-family:sans-serif">💎 Diamond Price Prediction App 💎</h1>
    <h4 style="color:#ffffff;text-align:center">Predict diamond prices with machine learning</h4> 
</div>
"""

# Deskripsi halaman Home
desc_temp = """ 
### Welcome to the Diamond Price Prediction App!
This app predicts diamond prices based on carat, cut, color, clarity, and dimensions.

#### Data Source
[Kaggle Diamond Dataset](https://www.kaggle.com/datasets/nancyalaswad90/diamonds-prices/data) 
"""

def main():
    stc.html(html_temp)
    menu = ["🏠 Home", "📊 Machine Learning App"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "🏠 Home":
        st.subheader("Home")
        st.markdown(desc_temp, unsafe_allow_html=True)
        st.image("https://cdn.pixabay.com/photo/2014/04/03/11/50/diamond-312822_1280.png", width=300)

    elif choice == "📊 Machine Learning App":
        run_ml_app()

    footer()

def run_ml_app():
    st.markdown("""
        <div style="padding:15px 0px">
            <h2 style="color:#00ffcc">Diamond Price Prediction</h2>
        </div>
    """, unsafe_allow_html=True)

    st.subheader("Please fill in the following details to predict diamond price")
    left, right = st.columns((2, 2))

    carat = left.number_input(label='Carat', min_value=0.001)
    cut = right.selectbox('Cut Quality', ('Fair', 'Good', 'Very Good', 'Premium', 'Ideal'))
    color = left.selectbox('Color (worst to best)', ('J', 'I', 'H', 'G', 'F', 'E', 'D'))
    clarity = right.selectbox('Clarity (worst to best)', ('I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'))
    depth = left.number_input(label='Total Depth Percentage', min_value=0.001)
    table = right.number_input(label='Table', min_value=0.001)
    x = left.number_input(label='Length (mm)', min_value=0.001)
    y = right.number_input(label='Width (mm)', min_value=0.001)
    z = left.number_input(label='Depth (mm)', min_value=0.001)

    button = st.button("🔮 Predict Diamond Price")

    if button:
        with st.spinner("Predicting... Please wait..."):
            result = predict(carat, cut, color, clarity, depth, table, x, y, z)
            st.success(f"The predicted price for this diamond is: ${result}")

def predict(carat, cut, color, clarity, depth, table, x, y, z):
    input = {
        'carat': carat,
        'cut': cut,
        'color': color,
        'clarity': clarity,
        'table': table,
        'x': x,
        'y': y,
        'z': z
    }
    df_predict = pd.DataFrame([input])

    # Encoding ordinal features
    df_predict[['cut', 'color', 'clarity']] = encoder.transform(df_predict[['cut', 'color', 'clarity']])

    # Reindex to match model input
    df_predict = df_predict.reindex(columns=feature_columns, fill_value=0)

    # Scaling
    df_predict = scaler.transform(df_predict[feature_columns])

    # Prediction
    log_prediction = XGBR_Model.predict(df_predict)
    result = np.expm1(log_prediction)[0]  # Inverse log transformation
    return round(result, 3)

def footer():
    st.markdown("""<hr style="margin-top:50px">
        <p style="text-align:center; color:gray;">
        © 2025 Diamond ML Predictor | Created with ❤️ using Streamlit
        </p>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
