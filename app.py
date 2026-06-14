import streamlit as st
import joblib
import re
import ssl # Tambahkan ini
import nltk
from nltk.corpus import stopwords as nltk_stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# --- INISIALISASI NLP & BYPASS SSL MAC ---
# Mengatasi error SSL Certificate Verify Failed pada Mac
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download stopwords jika belum ada
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

stopwords = set(nltk_stopwords.words('indonesian'))
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# --- FUNGSI LOAD MODEL ---
@st.cache_resource
def load_models():
    # Pastikan file model dan vectorizer Anda bernama sama
    model = joblib.load('model_svm_tfidf.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, vectorizer

# --- FUNGSI PREPROCESSING ---
def clean_tweet(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords]
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Sentimen Analisis Twitter", page_icon="🐦", layout="centered")

st.title("🐦 Dashboard Analisis Sentimen Twitter")

# Membuat dua Tab untuk merapikan tampilan sesuai 4 poin fitur minimal
tab1, tab2 = st.tabs(["🔍 Prediksi Sentimen", "ℹ️ Informasi Model & Evaluasi"])

with tab1:
    st.subheader("1. Prediksi Sentimen (Input & Hasil)")
    st.write("Masukkan teks tweet untuk mengetahui apakah sentimennya Positif, Negatif, atau Netral.")
    
    try:
        model, vectorizer = load_models()
    except FileNotFoundError:
        st.error("⚠️ File model_sentimen.pkl atau vectorizer.pkl tidak ditemukan!")
        st.stop()

    # Fitur 1: Input teks untuk prediksi sentimen
    user_input = st.text_area("Masukkan Teks Tweet:", height=150, placeholder="Contoh: Pelayanan di sini sangat buruk, saya kecewa!")

    if st.button("Analisis Sentimen", type="primary"):
        if user_input.strip() == "":
            st.warning("Silakan masukkan teks terlebih dahulu!")
        else:
            with st.spinner("Sedang memproses..."):
                # Preprocessing
                cleaned_text = clean_tweet(user_input)
                # Vectorizing
                text_vector = vectorizer.transform([cleaned_text])
                # Prediksi
                prediction = model.predict(text_vector)[0]
                
                st.markdown("---")
                # Fitur 2: Hasil prediksi sentimen
                st.write("**Hasil Prediksi:**")
                if prediction == 'Positif' or prediction == 1:
                    st.success("🌟 Sentimen: **POSITIF**")
                elif prediction == 'Negatif' or prediction == 0:
                    st.error("⚠️ Sentimen: **NEGATIF**")
                else:
                    st.info("⚖️ Sentimen: **NETRAL**")
                
                with st.expander("Lihat detail preprocessing teks"):
                    st.write(f"Teks bersih: `{cleaned_text}`")

with tab2:
    # Fitur 3: Informasi singkat mengenai dataset dan model
    st.subheader("Informasi Dataset & Model")
    st.markdown("""
    * **Dataset:** Dataset yang digunakan berasal dari Twitter (X) yang berisi kumpulan cuitan (tweet) berbahasa Indonesia dengan label sentimen umum, dengan total **10.806** baris data. Dataset telah melalui tahap pembersihan, penghapusan *stopword*, dan *stemming*.
    * **Model Machine Learning:** Model yang digunakan untuk klasifikasi adalah **Support Vector Machine**.
    * **Ekstraksi Fitur:** TF-IDF Vectorizer.
    """)
    
    st.divider()

    # Fitur 4: Hasil evaluasi model (accuracy atau metrik lainnya)
    st.subheader("Hasil Evaluasi Model")
    st.write("Berdasarkan proses pengujian (testing) yang telah dilakukan saat pelatihan, berikut adalah metrik performa model:")
    
    # Menggunakan st.columns untuk tampilan metrik yang lebih profesional
    col1, col2, col3 = st.columns(3)
    
    # SILAKAN UBAH ANGKA DI BAWAH INI SESUAI DENGAN HASIL TRAINING ANDA SEBENARNYA
    col1.metric(label="Accuracy", value="85.5%")
    col2.metric(label="Precision", value="84.2%")
    col3.metric(label="Recall", value="86.1%")
    
    st.info("Catatan: Metrik di atas dievaluasi menggunakan data *testing* sebesar [UBAH: persentase split, misal 20%].")