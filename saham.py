import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import streamlit as st

# Apply custom styles with a gradient background
st.markdown("""
    <style>
    .main { 
        background: linear-gradient(to right, #66ccff, #99ff99); 
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
option = st.sidebar.radio(
    "Choose a section", 
    ["Homepage", "Dataset Overview", "Data Visualization", "Model Training", "Make Prediction", "About Us"]
)

# Homepage
if option == "Homepage":
    st.title("Aplikasi Prediksi Harga Saham")
    st.subheader("Prediksi Harga Saham dengan Mudah dan Cepat")

    # Display an image on the homepage
    st.image("foto.jpg", caption="Analisis Pasar Saham", use_container_width=True)

    st.write("""
    Aplikasi ini dirancang untuk membantu Anda memprediksi harga saham berdasarkan data pasar. Dengan antarmuka yang interaktif dan visualisasi data yang informatif, Anda dapat:
    - Melihat data saham secara detail.
    - Menganalisis pola melalui visualisasi grafik.
    - Melatih model prediksi berbasis **Machine Learning**.
    - Membuat prediksi harga saham secara langsung.
    """)

    st.header("Fitur Utama")
    # Display a photo below "Fitur Utama"
    st.image("fitur.jpg", caption="Fitur Utama - Wawasan Pasar Saham", use_container_width=True)
    st.write("""
    1. **Dataset Overview**: Jelajahi data mentah dengan informasi yang lengkap dan mudah diakses.
    2. **Data Visualization**: Visualisasi data dengan histogram, heatmap, dan analisis lainnya.
    3. **Model Training**: Latih model prediksi harga saham menggunakan Random Forest Regressor.
    4. **Make Prediction**: Lakukan prediksi dengan memasukkan parameter seperti MarketCap dan Shares.
    """)

    st.header("Cara Menggunakan Aplikasi")
    # Display a photo below "Cara Menggunakan Aplikasi"
    st.image("aplikasi.jpg", caption="Cara Menggunakan Aplikasi", use_container_width=True)

    st.write("""
    1. Navigasikan ke bagian **Dataset Overview** untuk melihat dan memahami data.
    2. Gunakan **Data Visualization** untuk analisis pola dan hubungan antar variabel.
    3. Latih model prediksi di bagian **Model Training**.
    4. Gunakan fitur **Make Prediction** untuk membuat prediksi harga saham secara instan.
    """)

    st.write("**Mulai eksplorasi data Anda dan buat prediksi sekarang!** 🚀")

elif option == "About Us":
    st.title("Tentang Kami")
    
    # Penjelasan tentang saham
    st.header("Apa Itu Saham?")
    st.write("""
    Saham adalah instrumen keuangan yang menunjukkan kepemilikan atas suatu perusahaan. Dengan membeli saham, seorang investor memiliki bagian dari perusahaan tersebut. Saham diperdagangkan di pasar modal dan harga saham dipengaruhi oleh berbagai faktor, termasuk kinerja perusahaan, kondisi ekonomi, dan sentimen pasar.

    Saham sering digunakan oleh investor untuk memperoleh keuntungan melalui dua cara utama:
    - **Dividen**: Pembagian laba perusahaan kepada pemegang saham.
    - **Capital Gain**: Keuntungan yang diperoleh dari selisih harga jual dan beli saham.

    Investasi saham dapat memberikan hasil yang menguntungkan, namun juga memiliki risiko yang perlu dikelola dengan hati-hati.
    """)

    # Penjelasan tentang mengapa menggunakan dataset saham
    st.header("Mengapa Menggunakan Dataset Saham?")
    st.write("""
    Penggunaan dataset saham sangat penting dalam analisis pasar dan prediksi harga saham. Dengan memanfaatkan data historis yang mencakup informasi harga saham, kapitalisasi pasar, volume perdagangan, dan faktor lainnya, kita dapat mengidentifikasi pola yang tersembunyi, tren pasar, dan hubungan antar berbagai variabel.

    **Mengapa dataset saham penting?**
    - **Menganalisis Pola**: Data historis memungkinkan kita untuk melihat tren harga dan pola pergerakan pasar di masa lalu.
    - **Prediksi Harga**: Dengan menggunakan teknik machine learning, dataset saham membantu memprediksi harga saham di masa depan, yang sangat berharga bagi investor dan trader.
    - **Pengambilan Keputusan**: Analisis data saham dapat membantu investor membuat keputusan yang lebih terinformasi dan mengelola risiko investasi mereka dengan lebih baik.

    Dengan dataset saham yang akurat, kami dapat mengembangkan model prediktif yang memberikan wawasan yang lebih baik tentang bagaimana harga saham dapat bergerak.
    """)

    # Visi dan Misi
    st.header("Visi dan Misi Kami")
    
    st.subheader("Visi")
    st.write("""
    **Visi kami** adalah menjadi platform utama yang menyediakan alat prediksi harga saham yang canggih dan dapat diandalkan untuk para investor dan analis keuangan. Kami ingin memberdayakan individu dan organisasi dengan wawasan berbasis data yang memungkinkan mereka untuk membuat keputusan investasi yang lebih cerdas dan lebih tepat waktu.
    """)

    st.subheader("Misi")
    st.write("""
    **Misi kami** adalah untuk menyediakan solusi prediksi harga saham yang dapat diakses oleh semua orang dengan menggunakan teknologi machine learning yang canggih. Kami berfokus pada:
    
    1. **Mengumpulkan dan Menganalisis Data**: Menyediakan dataset saham yang lengkap dan akurat untuk membantu pengguna menganalisis pola pasar.
    2. **Membangun Model Prediksi yang Akurat**: Menggunakan algoritma machine learning untuk membangun model yang dapat memprediksi harga saham dengan akurasi tinggi.
    3. **Memberikan Wawasan yang Dapat Diterapkan**: Menyajikan informasi dan prediksi yang mudah dipahami dan dapat digunakan untuk mengambil keputusan investasi yang lebih baik.
    4. **Mengedukasi Pengguna**: Menyediakan alat dan sumber daya untuk mengedukasi pengguna tentang cara menggunakan data dan teknologi dalam investasi saham.
    """)

    st.write("""
    Kami berkomitmen untuk terus meningkatkan model dan fitur aplikasi kami, dengan tujuan memberikan solusi yang lebih baik dalam menganalisis dan memprediksi harga saham di masa depan. Terima kasih telah mempercayakan kami untuk membantu Anda dalam perjalanan investasi Anda!
    """)

#Dataset Overview
elif option == "Dataset Overview":
    st.title("Dataset Overview")
    
    st.write("""
    **Dataset Overview** memberikan gambaran umum tentang data yang digunakan dalam aplikasi prediksi harga saham. Dataset ini berisi informasi penting yang digunakan untuk melatih model dan membuat prediksi harga saham. Dataset ini sangat penting untuk analisis pasar saham, karena memberikan data historis yang memungkinkan pengguna untuk menganalisis tren harga saham, faktor-faktor yang mempengaruhi harga, dan hubungan antar variabel terkait.
    
    Beberapa fitur utama yang terkandung dalam dataset ini antara lain:
    - **MarketCap**: Kapitalisasi pasar perusahaan, yang mencerminkan ukuran dan nilai total perusahaan berdasarkan harga saham dan jumlah saham yang beredar. Ini memberikan gambaran umum tentang skala perusahaan dan potensinya di pasar saham.
    - **Shares**: Jumlah saham yang beredar di pasar, yang menunjukkan banyaknya saham yang dapat diperdagangkan oleh investor. Data ini penting karena mempengaruhi likuiditas dan volatilitas saham.
    - **ListingBoard**: Menunjukkan papan listing saham, yang membedakan pasar utama dan pasar sekunder. Misalnya, beberapa saham dapat terdaftar di pasar utama yang lebih besar atau pasar sekunder yang lebih kecil. Ini bisa mempengaruhi cara saham diperdagangkan dan volatilitas harga.
    - **Sector**: Sektor industri tempat perusahaan beroperasi, memberikan gambaran mengenai jenis bisnis perusahaan. Ini membantu investor mengidentifikasi sektor-sektor yang berkembang pesat atau yang sedang tertekan dalam perekonomian.
    - **LastPrice**: Harga saham terakhir yang tercatat pada perdagangan terakhir, yang digunakan untuk menentukan perubahan harga dan fluktuasi pasar. Informasi ini menjadi dasar untuk analisis pergerakan harga saham.
    - **ListingDate**: Tanggal perusahaan terdaftar di pasar saham, yang memberikan informasi tentang seberapa lama perusahaan tersebut diperdagangkan di pasar. Semakin lama perusahaan terdaftar, semakin stabil dan terbukti rekam jejak kinerjanya.
    
    Dataset ini digunakan untuk melakukan analisis pola dan prediksi harga saham dengan menggunakan algoritma machine learning. Dengan dataset yang akurat dan lengkap, model prediksi dapat memberikan estimasi harga saham di masa depan yang lebih tepat, membantu investor dalam pengambilan keputusan investasi yang lebih informasional dan tepat waktu.
    """)
    
    try:
        # Memuat dataset
        data_overview = pd.read_csv("stock_data.csv")  # Dataset khusus untuk Dataset Overview
        st.write("### Dataset Loaded Successfully!")
        st.dataframe(data_overview)  # Menampilkan dataset dalam format tabel interaktif
        
        st.write("""
        **Penjelasan Dataset**: 
        Dataset ini berisi informasi tentang harga saham dan fitur terkait yang dapat digunakan untuk melakukan analisis dan prediksi. Dataset ini terdiri dari berbagai variabel numerik dan kategorikal yang memberikan wawasan mendalam tentang faktor-faktor yang mempengaruhi harga saham. Dengan menggunakan dataset ini, kita dapat mengidentifikasi hubungan antara kapitalisasi pasar, sektor, volume perdagangan, dan harga saham. Informasi ini sangat berguna untuk memperkirakan bagaimana harga saham dapat bergerak di masa depan.
        
        Selain itu, analisis dataset saham memungkinkan kita untuk memahami faktor-faktor yang mempengaruhi pergerakan pasar secara lebih baik, seperti sektor-sektor yang sedang berkembang atau tertekan, serta tren pasar yang dapat dimanfaatkan untuk membuat keputusan investasi yang lebih baik.
        """)
        
        # Menampilkan Info mengenai dataset
        st.write("### Data Info")
        st.write(data_overview.info())  # Struktur dataset: jumlah baris, kolom, tipe data, dll.
        
        st.write("""
        **Data Info** memberikan informasi tentang struktur dataset, termasuk jumlah baris, kolom, dan tipe data dari masing-masing kolom. Ini sangat penting untuk memahami komposisi data dan jenis analisis yang dapat dilakukan berdasarkan tipe data yang ada. Misalnya, kita dapat melihat apakah ada kolom yang berisi data numerik yang dapat langsung digunakan untuk model prediksi, atau kolom yang berisi data kategorikal yang memerlukan encoding sebelum digunakan.
        
        Memahami tipe data juga penting untuk memastikan bahwa data telah diproses dengan benar sebelum analisis lebih lanjut, serta untuk mendeteksi adanya kesalahan atau inkonsistensi dalam dataset.
        """)
        
        # Menampilkan deskripsi statistik dataset
        st.write("### Data Description")
        st.write(data_overview.describe())  # Statistik deskriptif: rata-rata, standar deviasi, dll.
        
        st.write("""
        **Data Description** menunjukkan statistik deskriptif dari data numerik dalam dataset, seperti rata-rata, standar deviasi, nilai minimum dan maksimum, serta kuartil. Statistik ini membantu untuk memahami sebaran data, kecenderungan nilai, serta variabilitas antar nilai dalam dataset.
        
        Sebagai contoh, kita dapat melihat apakah ada variabel yang memiliki rentang nilai yang sangat besar, atau apakah ada variabel yang cenderung terpusat pada nilai tertentu. Informasi ini sangat penting untuk menentukan langkah-langkah selanjutnya dalam analisis, seperti normalisasi atau transformasi data, yang dapat meningkatkan akurasi model prediksi.
        """)
        
        # Memeriksa nilai yang hilang (Missing Values)
        st.write("### Missing Values")
        st.write(data_overview.isnull().sum())  # Memeriksa nilai yang hilang di setiap kolom

        st.write("""
        **Missing Values** mengidentifikasi apakah terdapat data yang hilang dalam dataset. Memeriksa nilai yang hilang penting untuk memastikan bahwa dataset bersih dan siap digunakan untuk analisis lebih lanjut. Nilai yang hilang dapat mempengaruhi hasil analisis dan model prediksi, sehingga penting untuk menangani nilai yang hilang sebelum melanjutkan analisis.
        
        Berbagai teknik dapat digunakan untuk menangani nilai yang hilang, seperti menghapus baris atau kolom yang mengandung nilai hilang, atau mengisi nilai hilang dengan nilai rata-rata atau median, tergantung pada jenis data dan pengaruhnya terhadap model.
        """)
        
    except FileNotFoundError:
        # Menangani error jika file tidak ditemukan
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

        # Stock Price by ListingBoard using Altair with annotations
        st.write("### Stock Price by ListingBoard with Annotations")
        
        # Create the boxplot
        chart = alt.Chart(data_viz).mark_boxplot().encode(
            x='ListingBoard:N',
            y='LastPrice:Q',
            color='ListingBoard:N',
            tooltip=['ListingBoard', 'LastPrice']
        ).properties(width=800, height=400)

        # Add annotations (e.g., showing the median of each boxplot)
        annotations = alt.Chart(data_viz).mark_text(
            align='center',
            baseline='middle',
            color='black',
            fontSize=12
        ).encode(
            x='ListingBoard:N',
            y=alt.Y('median(LastPrice):Q', title='Median Price'),
            text='median(LastPrice):Q'
        )

        # Combine the boxplot and annotations
        st.altair_chart(chart + annotations, use_container_width=True)

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
# Make Prediction Section
elif option == "Make Prediction":
    st.title("Make Predictions")
    
    # Create select sliders for prediction
    col1, col2 = st.columns(2)
    with col1:
        market_cap = st.select_slider("MarketCap:", options=[100000, 200000, 300000, 400000, 500000])
        shares = st.select_slider("Shares:", options=[1000, 5000, 10000, 15000, 20000])

    with col2:
        listing_board = st.select_slider("ListingBoard (as numeric):", options=[0, 1, 2, 3, 4])
        sector = st.select_slider("Sector (as numeric):", options=[0, 1, 2, 3, 4])

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
