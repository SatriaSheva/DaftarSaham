import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import streamlit as st

# Apply custom styles
st.markdown("""
    <style>
    .main { 
        background-color: #f0f8ff; 
        color: #333;
    }
    h1 {
        font-family: 'Arial', sans-serif;
        color: #004d99;
    }
    h2 {
        font-family: 'Arial', sans-serif;
        color: #0066cc;
    }
    .sidebar .sidebar-content {
        background-color: #e6f7ff;
    }
    .stButton>button {
        background-color: #0066cc;
        color: white;
        font-weight: bold;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #005bb5;
    }
    </style>
    """, unsafe_allow_html=True)

# Load Dataset
st.title("Stock Price Prediction Application")

try:
    # Load CSV File
    data = pd.read_csv("stock_data.csv")
    st.write("### Dataset Loaded Successfully!")
except FileNotFoundError:
    st.error("File `stock_data.csv` not found! Please ensure the file is in the same directory as this script.")
    st.stop()

# Sidebar for Navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio("Choose a section", ["Dataset Overview", "Data Visualization", "Model Training", "Make Prediction"])

# Dataset Overview Section
if option == "Dataset Overview":
    st.write("### Full Dataset")
    st.dataframe(data)  # Display the entire dataset interactively

    st.write("### Data Info")
    st.write(data.info())

    st.write("### Data Description")
    st.write(data.describe())

    st.write("### Missing Values")
    st.write(data.isnull().sum())

# Data Visualization Section
elif option == "Data Visualization":
    st.write("### Stock Price Distribution")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(data['LastPrice'], bins=30, kde=True, color='blue', ax=ax)
    st.pyplot(fig)

    st.write("### Correlation Heatmap")
    correlation = data[['LastPrice', 'MarketCap', 'Shares']].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)

    st.write("### Stock Price by ListingBoard")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(x='ListingBoard', y='LastPrice', data=data, ax=ax)
    st.pyplot(fig)

# Model Training Section
elif option == "Model Training":
    # Convert ListingDate to datetime
    data['ListingDate'] = pd.to_datetime(data['ListingDate'], errors='coerce')

    # Data Preprocessing
    data['ListingBoard'] = data['ListingBoard'].astype('category').cat.codes
    data['Sector'] = data['Sector'].astype('category').cat.codes

    # Feature and Target Selection
    st.write("### Preprocessing and Training")
    X = data[['MarketCap', 'Shares', 'ListingBoard', 'Sector']]
    y = data['LastPrice']

    # Validate Data
    X = X.apply(pd.to_numeric, errors='coerce')  # Convert to numeric
    y = pd.to_numeric(y, errors='coerce')  # Convert to numeric
    X = X.dropna()  # Drop rows with NaN
    y = y[X.index]  # Align target with features

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Show progress bar while training
    progress_bar = st.progress(0)
    st.write("Training Random Forest Model...")

    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Update progress bar
    progress_bar.progress(100)

    # Evaluate Model
    y_pred = rf_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    st.write(f"**Mean Absolute Error (MAE):** {mae}")
    st.write(f"**Root Mean Squared Error (RMSE):** {rmse}")

    # Save model
    joblib.dump(rf_model, 'rf_model.pkl')
    st.write("Model training complete and saved as `rf_model.pkl`")

# Make Prediction Section
elif option == "Make Prediction":
    st.write("### Make Predictions")
    
    # Create input fields for prediction
    col1, col2 = st.columns(2)
    with col1:
        market_cap = st.number_input("MarketCap:", min_value=0.0, step=1.0)
        shares = st.number_input("Shares:", min_value=0.0, step=1.0)
    
    with col2:
        listing_board = st.number_input("ListingBoard (as numeric):", min_value=0, step=1)
        sector = st.number_input("Sector (as numeric):", min_value=0, step=1)

    # Load the trained model
    try:
        rf_model = joblib.load('rf_model.pkl')
    except FileNotFoundError:
        st.error("Model not found! Please train the model first.")
        st.stop()

    if st.button("Predict Stock Price"):
        input_data = pd.DataFrame({
            'MarketCap': [market_cap],
            'Shares': [shares],
            'ListingBoard': [listing_board],
            'Sector': [sector]
        })
        prediction = rf_model.predict(input_data)[0]
        st.write(f"**Predicted Stock Price:** {prediction:.2f}", unsafe_allow_html=True)
