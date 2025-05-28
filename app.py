import streamlit as st
import streamlit.components.v1 as stc
import pickle
import pandas as pd
import numpy as np
from sklearn import preprocessing, sklearn
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

with open('lgbr_model.pkl', 'rb') as file:
    LGBR_Model = pickle.load(file)
    
with open('feature_columns.pkl', 'rb') as file:
    feature_columns = pickle.load(file)
    
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)

html_temp = """<div style="background-color:#000;padding:10px;border-radius:10px">
                <h1 style="color:#fff;text-align:center">Diamond Price Prediction App</h1> 
                <h4 style="color:#fff;text-align:center">_____insert something here_____</h4> 
                """

desc_temp = """ ### Diamond Price Prediction App 
                This app is used for predicting diamond prices based on various features such as carat, cut, color, clarity, and dimensions.
                
                #### Data Source
                Kaggle: Link https://www.kaggle.com/datasets/nancyalaswad90/diamonds-prices/data 
                """

def main():
    stc.html(html_temp)
    menu = ["Home", "Machine Learning App"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Home")
        st.markdown(desc_temp, unsafe_allow_html=True)
    elif choice == "Machine Learning App":
        run_ml_app()

def run_ml_app():
    design = """<div style="padding:15px;">
                    <h1 style="color:#fff">Diamond Price Prediction</h1>
                </div
             """
    st.markdown(design, unsafe_allow_html=True)

    # Create form structure
    st.subheader("Please fill in the following details to predict diamond price")
    left,right = st.columns((2,2))
    carat = left.number_input(label = 'Carat', min_value=0.001)
    cut = right.selectbox('Cut Quality',('Fair','Good','Very Good','Premium','Ideal'))
    color = left.selectbox('Color',('J','I','H','G','F','E','D'))
    clarity = right.selectbox('Clarity',('I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF'))
    depth = left.number_input(label = 'Total Depth Percentage', min_value=0.001)
    table = right.number_input(label = 'Table', min_value=0.001)
    x = left.number_input(label = 'Length (mm)', min_value=0.001)
    y = right.number_input(label = 'Width (mm)', min_value=0.001)
    z = left.number_input(label = 'Depth (mm)', min_value=0.001)
    button = st.button("Predict Diamond Price")
    #If button is clicked
    if button:
        result = predict(carat, cut, color, clarity, depth, table, x, y, z)
        st.success(f"The predicted price for this diamond is: ${result}")

def predict(carat, cut, color, clarity, depth, table, x, y, z):
    #Processing user input
    input = {
        'carat': carat,
        'cut': cut,
        'color': color,
        'clarity': clarity,
        'depth': depth,
        'table': table,
        'x': x,
        'y': y,
        'z': z
    }
    df_predict = pd.DataFrame([input])
    df_predict = df_predict.drop(columns=['depth', 'table'], errors='ignore')
    
    # Encoding ordinal features
    df_predict[['cut', 'color', 'clarity']] = encoder.fit_transform(
        df_predict[['cut', 'color', 'clarity']])
    
    # Reindex to match the model's expected input
    df_predict = df_predict.reindex(columns=feature_columns, fill_value=0)

    # Scaling the features
    df_predict = scaler.transform(df_predict[feature_columns])

    #Making prediction
    log_prediction = LGBR_Model.predict(df_predict)
    result = np.expm1(log_prediction)[0]  # Inverse log transformation
    # Round result to 3 decimal places
    return round(result,3)

if __name__ == "__main__":
    main()