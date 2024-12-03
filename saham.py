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

# Homepage
st.sidebar.title("Navigation")
option = st.sidebar.radio(
    "Choose a section", 
    ["Homepage", "Dataset Overview", "Data Visualization", "Model Training", "Make Prediction"]
)

if option == "Homepage":
    st.title("Stock Price Prediction Application")
    st.subheader("Prediksi Harga Saham dengan Mudah dan Cepat")

    st.write("""
    Aplikasi ini dirancang untuk membantu Anda memprediksi harga saham berdasarkan data pasar. Dengan antarmuka yang interaktif dan visualisasi data yang informatif, Anda dapat:
    - Melihat data saham secara detail.
    - Menganalisis pola melalui visualisasi grafik.
    - Melatih model prediksi berbasis **Machine Learning**.
    - Membuat prediksi harga saham secara langsung.
    """)

    st.header("Fitur Utama")
    st.write("""
    1. **Dataset Overview**: Jelajahi data mentah dengan informasi yang lengkap dan mudah diakses.
    2. **Data Visualization**: Visualisasi data dengan histogram, heatmap, dan analisis lainnya.
    3. **Model Training**: Latih model prediksi harga saham menggunakan Random Forest Regressor.
    4. **Make Prediction**: Lakukan prediksi dengan memasukkan parameter seperti MarketCap dan Shares.
    """)

    st.header("Cara Menggunakan Aplikasi")
    st.write("""
    1. Navigasikan ke bagian **Dataset Overview** untuk melihat dan memahami data.
    2. Gunakan **Data Visualization** untuk analisis pola dan hubungan antar variabel.
    3. Latih model prediksi di bagian **Model Training**.
    4. Gunakan fitur **Make Prediction** untuk membuat prediksi harga saham secara instan.
    """)

    st.write("**Mulai eksplorasi data Anda dan buat prediksi sekarang!** 🚀")

# Dataset Overview Section
elif option == "Dataset Overview":
    st.title("Dataset Overview")
    try:
        data_overview = pd.read_csv("stock_data.csv")  # Dataset khusus untuk Dataset Overview
        st.write("### Dataset Loaded Successfully!")
        st.dataframe(data_overview)  # Display the entire dataset interactively
        st.write("### Data Info")
        st.write(data_overview.info())
        st.write("### Data Description")
        st.write(data_overview.describe())
        st.write("### Missing Values")
        st.write(data_overview.isnull().sum())
    except FileNotFoundError:
        st.error("File `stock_data.csv` not found! Please ensure the file is in the same directory as this script.")
        st.stop()

# Data Visualization Section
elif option == "Data Visualization":
    st.title("Data Visualization")
    try:
        data_viz = pd.read_csv("stock_data.csv")  # Dataset khusus untuk Data Visualization
        st.write("### Stock Price Distribution")
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data_viz['LastPrice'], bins=30, kde=True, color='blue', ax=ax)
        st.pyplot(fig)

        st.write("### Correlation Heatmap")
        correlation = data_viz[['LastPrice', 'MarketCap', 'Shares']].corr()
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
        st.pyplot(fig)

        st.write("### Stock Price by ListingBoard")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(x='ListingBoard', y='LastPrice', data=data_viz, ax=ax)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Model Training Section
elif option == "Model Training":
    st.title("Model Training")
    try:
        data_train = pd.read_csv("stock_data.csv")  # Dataset khusus untuk Model Training
        data_train['ListingDate'] = pd.to_datetime(data_train['ListingDate'], errors='coerce')

        # Data Preprocessing
        data_train['ListingBoard'] = data_train['ListingBoard'].astype('category').cat.codes
        data_train['Sector'] = data_train['Sector'].astype('category').cat.codes

        # Feature and Target Selection
        st.write("### Preprocessing and Training")
        X = data_train[['MarketCap', 'Shares', 'ListingBoard', 'Sector']]
        y = data_train['LastPrice']

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
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Make Prediction Section
elif option == "Make Prediction":
    st.title("Make Predictions")
    
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
        st.write(f"**Predicted Stock Price:** {prediction:.2f}")
