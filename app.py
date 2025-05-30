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

# Custom CSS for elegant styling
st.markdown("""
    <style>
        .main {
            background-color: #0E1117;
        }
        .stApp {
            background: linear-gradient(135deg, #0a0a2a 0%, #1a1a4a 100%);
            color: #ffffff;
        }
        .sidebar .sidebar-content {
            background: linear-gradient(195deg, #0a0a2a 0%, #1a1a4a 100%) !important;
            border-right: 1px solid #4a4a8a;
        }
        .stButton>button {
            background: linear-gradient(90deg, #4b6cb7 0%, #182848 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 12px 24px;
            font-weight: bold;
            transition: all 0.3s ease;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 8px rgba(0, 0, 0, 0.15);
            background: linear-gradient(90deg, #5b7cc7 0%, #283858 100%);
        }
        .stNumberInput, .stSelectbox {
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            padding: 8px;
            border: 1px solid #4a4a8a;
        }
        .stNumberInput input, .stSelectbox select {
            color: white !important;
        }
        .stSuccess {
            background-color: rgba(0, 200, 83, 0.2) !important;
            border: 1px solid #00c853 !important;
            border-radius: 8px;
            padding: 15px;
            font-size: 18px;
            text-align: center;
        }
        .diamond-icon {
            color: #b9f2ff;
            text-shadow: 0 0 10px #b9f2ff, 0 0 20px #b9f2ff;
        }
        .header-text {
            background: linear-gradient(90deg, #b9f2ff 0%, #ffffff 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 0 5px rgba(185, 242, 255, 0.3);
        }
        .feature-card {
            background: rgba(20, 20, 60, 0.7);
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            border-left: 4px solid #4b6cb7;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
    </style>
""", unsafe_allow_html=True)

# HTML templates
html_temp = """
<div style="background: linear-gradient(135deg, #0a0a2a 0%, #1a1a4a 100%);
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.3);
            border: 1px solid #4a4a8a;
            text-align: center;
            margin-bottom: 30px;">
    <h1 style="color: #b9f2ff; 
               font-size: 2.5em; 
               margin-bottom: 10px;
               text-shadow: 0 0 10px rgba(185, 242, 255, 0.5);">
        <span class="diamond-icon">◆</span> Diamond Price Prediction <span class="diamond-icon">◆</span>
    </h1>
    <h4 style="color: #b9f2ff; 
               font-weight: 300;
               letter-spacing: 1px;">
        Discover the true value of your precious gem
    </h4>
</div>
"""

desc_temp = """
<div class="feature-card">
    <h2 class="header-text">✨ Diamond Price Prediction App</h2>
    <p style="color: #d1d1e0; line-height: 1.6;">
        This sophisticated tool helps you estimate the market value of diamonds based on their unique characteristics. 
        Our advanced machine learning model analyzes carat weight, cut quality, color grade, clarity, and dimensions 
        to provide an accurate price prediction.
    </p>
</div>

<div class="feature-card">
    <h3 class="header-text">🔍 How It Works</h3>
    <p style="color: #d1d1e0; line-height: 1.6;">
        1. Navigate to the <b>Machine Learning App</b> section<br>
        2. Enter your diamond's specifications<br>
        3. Click "Predict Diamond Price"<br>
        4. Receive an instant valuation<br>
    </p>
</div>

<div class="feature-card">
    <h3 class="header-text">💎 Data Source</h3>
    <p style="color: #d1d1e0; line-height: 1.6;">
        Our model was trained on a comprehensive dataset from Kaggle:<br>
        <a href="https://www.kaggle.com/datasets/nancyalaswad90/diamonds-prices/data" 
           style="color: #b9f2ff; text-decoration: none;">
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
    design = """
    <div style="background: linear-gradient(135deg, rgba(10, 10, 42, 0.8) 0%, rgba(26, 26, 74, 0.8) 100%);
                padding: 20px;
                border-radius: 15px;
                box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
                margin-bottom: 30px;
                border: 1px solid #4a4a8a;">
        <h1 style="color: #b9f2ff; 
                   text-align: center;
                   font-size: 2em;
                   margin-bottom: 0;">
            Diamond Valuation Tool
        </h1>
        <p style="color: #d1d1e0; 
                  text-align: center;
                  font-size: 1em;
                  margin-top: 5px;">
            Enter your diamond's specifications below
        </p>
    </div>
    """
    st.markdown(design, unsafe_allow_html=True)

    # Create form structure with elegant layout
    st.subheader("Diamond Specifications", divider="rainbow")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Basic Attributes")
        carat = st.number_input('Carat Weight', min_value=0.001, max_value=10.0, value=1.0, step=0.01,
                              help="Weight of the diamond in carats")
        cut = st.selectbox('Cut Quality', ('Fair', 'Good', 'Very Good', 'Premium', 'Ideal'),
                          help="Quality of the diamond's cut")
        color = st.selectbox('Color Grade (D = best, J = worst)', ('D', 'E', 'F', 'G', 'H', 'I', 'J'),
                            help="Color grade of the diamond")
        
    with col2:
        st.markdown("### Quality Metrics")
        clarity = st.selectbox('Clarity Grade', ('IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1', 'SI2', 'I1'),
                             help="Clarity grade of the diamond")
        depth = st.number_input('Total Depth Percentage', min_value=0.001, max_value=100.0, value=60.0, step=0.1,
                              help="Total depth percentage (depth/mean diameter)")
        table = st.number_input('Table Width', min_value=0.001, max_value=100.0, value=55.0, step=0.1,
                              help="Width of the diamond's table")
    
    st.markdown("### Dimensions (in millimeters)")
    dim_col1, dim_col2, dim_col3 = st.columns(3)
    with dim_col1:
        x = st.number_input('Length (X)', min_value=0.001, value=5.0, step=0.01,
                           help="Length of the diamond in mm")
    with dim_col2:
        y = st.number_input('Width (Y)', min_value=0.001, value=5.0, step=0.01,
                           help="Width of the diamond in mm")
    with dim_col3:
        z = st.number_input('Depth (Z)', min_value=0.001, value=3.0, step=0.01,
                           help="Depth of the diamond in mm")
    
    # Centered predict button with diamond icon
    st.markdown("<div style='text-align: center; margin: 30px 0;'>", unsafe_allow_html=True)
    button = st.button("✨ Predict Diamond Price ✨", key="predict_button")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # If button is clicked
    if button:
        with st.spinner('Calculating value...'):
            result = predict(carat, cut, color, clarity, depth, table, x, y, z)
            st.success(f"The estimated value for this diamond is: **${result:,.2f}**")
            st.balloons()

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
