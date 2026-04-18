import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("🏨 Hotel Recommender")

# 1. Veriyi Yükle
df = joblib.load('hotel_data_package.pkl')

# 2. Girişler
country = st.selectbox("Select Country:", df['Countries'].unique())
desc = st.text_input("Describe your trip:", "business trip")

if st.button("Find"):
    # Filtrele ve Dinamik Analiz Yap
    sub_df = df[df['Countries'].str.contains(country, case=False)].copy()
    
    if not sub_df.empty:
        tf = TfidfVectorizer(stop_words='english')
        # Matrisi oluştur ve benzerliği hesapla
        matrix = tf.fit_transform(sub_df['Features'])
        scores = cosine_similarity(matrix, tf.transform([desc])).flatten()
        
        # En iyi 3 sonucu getir
        results = sub_df['Hotel_Name'].iloc[scores.argsort()[-3:][::-1]]
        
        for hotel in results:
            st.success(f"🏢 {hotel}")
    else:
        st.error("No hotels found.")