import streamlit as st
import streamlit.components.v1 as stc
import pickle
import pandas as pd
import numpy as np
from sklearn import preprocessing, sklearn
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# Load models and preprocessing objects
with open('xgbr_model.pkl', 'rb') as file:
    XGBR_Model = pickle.load(file)
    
with open('feature_columns.pkl', 'rb') as file:
    feature_columns = pickle.load(file)
    
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

with open('encoder.pkl', 'rb') as file:
    encoder = pickle.load(file)

# Custom CSS for clean, bright styling
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .stApp {
            background-color: #ffffff;
        }
        .sidebar .sidebar-content {
            background-color: #ffffff !important;
            border-right: 1px solid #e0e0e0;
        }
        .stButton>button {
            background-color: #4b6cb7;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 12px 24px;
            font-weight: bold;
            transition: all 0.3s ease;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            width: 100%;
            font-size: 16px;
        }
        .stButton>button:hover {
            background-color: #3a5a9c;
            transform: translateY(-1px);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .stNumberInput, .stSelectbox {
            background-color: #ffffff;
            border-radius: 8px;
            border: 1px solid #ced4da;
        }
        .stNumberInput input, .stSelectbox select {
            color: #495057 !important;
        }
        .stSuccess {
            background-color: rgba(0, 200, 83, 0.1) !important;
            border: 1px solid #00c853 !important;
            border-radius: 8px;
            padding: 15px;
            font-size: 18px;
            text-align: center;
            color: #00c853 !important;
        }
        .header-box {
            background-color: #f1f3f5;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            border-left: 5px solid #4b6cb7;
        }
        .section-title {
            color: #4b6cb7;
            font-weight: 600;
            margin-bottom: 15px;
        }
        .input-label {
            font-weight: 500;
            color: #495057;
            margin-bottom: 5px;
        }
        .input-description {
            font-size: 12px;
            color: #6c757d;
            margin-top: 3px;
        }
    </style>
""", unsafe_allow_html=True)

# HTML templates
html_temp = """
<div class="header-box">
    <h1 style="color: #2c3e50; text-align: center; margin-bottom: 5px;">
        Diamond Price Prediction
    </h1>
    <p style="color: #6c757d; text-align: center; font-size: 16px;">
        Estimate the market value of your diamond with precision
    </p>
</div>
"""

desc_temp = """
<div style="background-color: #f8f9fa; border-radius: 10px; padding: 20px; margin-bottom: 20px;">
    <h3 style="color: #2c3e50;">About This App</h3>
    <p style="color: #495057;">
        This tool helps you estimate the market value of diamonds based on their unique characteristics. 
        Our advanced machine learning model analyzes various diamond attributes to provide an accurate price prediction.
    </p>
</div>

<div style="background-color: #f8f9fa; border-radius: 10px; padding: 20px; margin-bottom: 20px;">
    <h3 style="color: #2c3e50;">How To Use</h3>
    <ol style="color: #495057;">
        <li>Navigate to the <b>Price Prediction</b> section</li>
        <li>Enter your diamond's specifications</li>
        <li>Click "Predict Diamond Price"</li>
        <li>View your estimated diamond value</li>
    </ol>
</div>

<div style="background-color: #f8f9fa; border-radius: 10px; padding: 20px;">
    <h3 style="color: #2c3e50;">Data Source</h3>
    <p style="color: #495057;">
        Our model was trained on a comprehensive dataset from Kaggle:<br>
        <a href="https://www.kaggle.com/datasets/nancyalaswad90/diamonds-prices/data" 
           style="color: #4b6cb7; text-decoration: none;">
           Diamond Prices Dataset
        </a>
    </p>
</div>
"""

def main():
    stc.html(html_temp)
    menu = ["Home", "Price Prediction"]
    choice = st.sidebar.selectbox("Navigation", menu, key="menu_select")

    if choice == "Home":
        st.markdown(desc_temp, unsafe_allow_html=True)
    elif choice == "Price Prediction":
        run_ml_app()

def run_ml_app():
    st.markdown("""
    <div style="background-color: #f1f3f5; border-radius: 10px; padding: 15px 20px; margin-bottom: 25px;">
        <h2 style="color: #2c3e50; margin: 0;">Diamond Specifications</h2>
        <p style="color: #6c757d; margin: 5px 0 0 0;">
            Please enter all required details about your diamond
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Create form structure with clear sections
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<p class="section-title">Basic Attributes</p>', unsafe_allow_html=True)
        
        st.markdown('<p class="input-label">Carat Weight</p>', unsafe_allow_html=True)
        carat = st.number_input('', min_value=0.001, max_value=10.0, value=1.0, step=0.01, 
                               key='carat_input', label_visibility="collapsed")
        st.markdown('<p class="input-description">Weight of the diamond in carats (1 carat = 0.2 grams)</p>', unsafe_allow_html=True)
        
        st.markdown('<p class="input-label">Cut Quality</p>', unsafe_allow_html=True)
        cut = st.selectbox('', ('Fair', 'Good', 'Very Good', 'Premium', 'Ideal'),
                          key='cut_select', label_visibility="collapsed")
        st.markdown('<p class="input-description">Quality of the diamond cut (Ideal is best)</p>', unsafe_allow_html=True)
        
        st.markdown('<p class="input-label">Color Grade</p>', unsafe_allow_html=True)
        color = st.selectbox('', ('D', 'E', 'F', 'G', 'H', 'I', 'J'),
                            key='color_select', label_visibility="collapsed")
        st.markdown('<p class="input-description">D (colorless) to J (light color)</p>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<p class="section-title">Quality Metrics</p>', unsafe_allow_html=True)
        
        st.markdown('<p class="input-label">Clarity Grade</p>', unsafe_allow_html=True)
        clarity = st.selectbox('', ('IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1'),
                             key='clarity_select', label_visibility="collapsed")
        st.markdown('<p class="input-description">IF (flawless) to I1 (visible inclusions)</p>', unsafe_allow_html=True)
        
        st.markdown('<p class="input-label">Depth Percentage</p>', unsafe_allow_html=True)
        depth = st.number_input('', min_value=0.001, max_value=100.0, value=60.0, step=0.1,
                              key='depth_input', label_visibility="collapsed")
        st.markdown('<p class="input-description">Total depth percentage (depth/mean diameter)</p>', unsafe_allow_html=True)
        
        st.markdown('<p class="input-label">Table Width</p>', unsafe_allow_html=True)
        table = st.number_input('', min_value=0.001, max_value=100.0, value=55.0, step=0.1,
                              key='table_input', label_visibility="collapsed")
        st.markdown('<p class="input-description">Width of diamond\'s table (top flat surface)</p>', unsafe_allow_html=True)
    
    st.markdown('<p class="section-title">Dimensions (in millimeters)</p>', unsafe_allow_html=True)
    dim_col1, dim_col2, dim_col3 = st.columns(3)
    with dim_col1:
        st.markdown('<p class="input-label">Length (X)</p>', unsafe_allow_html=True)
        x = st.number_input('', min_value=0.001, value=5.0, step=0.01,
                           key='x_input', label_visibility="collapsed")
        st.markdown('<p class="input-description">Length dimension in mm</p>', unsafe_allow_html=True)
    with dim_col2:
        st.markdown('<p class="input-label">Width (Y)</p>', unsafe_allow_html=True)
        y = st.number_input('', min_value=0.001, value=5.0, step=0.01,
                           key='y_input', label_visibility="collapsed")
        st.markdown('<p class="input-description">Width dimension in mm</p>', unsafe_allow_html=True)
    with dim_col3:
        st.markdown('<p class="input-label">Depth (Z)</p>', unsafe_allow_html=True)
        z = st.number_input('', min_value=0.001, value=3.0, step=0.01,
                           key='z_input', label_visibility="collapsed")
        st.markdown('<p class="input-description">Depth dimension in mm</p>', unsafe_allow_html=True)
    
    # Predict button
    st.markdown("<div style='margin: 30px 0;'>", unsafe_allow_html=True)
    button = st.button("Predict Diamond Price", key="predict_button")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # If button is clicked
    if button:
        with st.spinner('Calculating value...'):
            result = predict(carat, cut, color, clarity, depth, table, x, y, z)
            st.success(f"The estimated value for this diamond is: **${result:,.2f}**")

def predict(carat, cut, color, clarity, depth, table, x, y, z):
    # Processing user input
    input_data = {
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
    df_predict = pd.DataFrame([input_data])
    df_predict = df_predict.drop(columns=['depth'], errors='ignore')
    
    # Encoding ordinal features
    df_predict[['cut', 'color', 'clarity']] = encoder.fit_transform(
        df_predict[['cut', 'color', 'clarity']])
    
    # Reindex to match the model's expected input
    df_predict = df_predict.reindex(columns=feature_columns, fill_value=0)

    # Scaling the features
    df_predict = scaler.transform(df_predict[feature_columns])

    # Making prediction
    log_prediction = XGBR_Model.predict(df_predict)
    result = np.expm1(log_prediction)[0]  # Inverse log transformation
    
    # Round result to 2 decimal places for currency
    return round(result, 2)

if __name__ == "__main__":
    main()
