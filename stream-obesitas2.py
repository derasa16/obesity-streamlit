import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
import os

st.write("Starting application...")

try:
    st.write("All modules imported successfully!")
    
    # Membaca model
    diabetes_model = pickle.load(open('obesitas_model.sav', 'rb'))

    # Membaca scaler
    scaler = pickle.load(open('StandardScaler.pkl', 'rb'))
    
    st.write("Model and scaler loaded successfully!")
    
except FileNotFoundError as e:
    st.error(f"FileNotFoundError: {e}")
except Exception as e:
    st.error(f"An error occurred: {e}")

# Judul web
st.title("Prediksi Tingkat Obesitas")
col1, col2 = st.columns(2)

with col1:
    Age = st.text_input("Usia")
    if Age != '':
        Age = float(Age)  # Konversi ke float

with col2:
    Height = st.text_input("Tinggi Badan")
    if Height != '':
        Height = float(Height)  # Konversi ke float

with col1:
    Weight = st.text_input("Berat Badan")
    if Weight != '':
        Weight = float(Weight)  # Konversi ke float

Sex_input = st.selectbox(
    "Pilih jenis kelamin:",
    ('Laki-Laki', 'Perempuan')
)
gender_mapping = {'Laki-Laki': 1, 'Perempuan': 0}
Sex_y = gender_mapping[Sex_input]

CALC_input = st.selectbox(
    "Seberapa Sering Mengkonsumsi Alkohol:",
    ('Tidak Pernah', 'Kadang-Kadang', 'Sering', 'Selalu')
)
CALC_mapping = {'Selalu': 3, 'Sering': 2, 'Kadang-Kadang': 1, 'Tidak Pernah': 0}
CALC_y = CALC_mapping[CALC_input]

FAVC_input = st.selectbox(
    "Apakah Anda Sering Mengkonsumsi Makanan Tinggi Kalori:",
    ('ya', 'tidak')
)
FAVC_mapping = {'ya': 1, 'tidak': 0}
FAVC_y = FAVC_mapping[FAVC_input]

FCVC_input = st.selectbox(
    "Seberapa Sering Mengkonsumsi Sayuran:",
    ('Tidak Pernah', 'Kadang-Kadang', 'Sering', 'Selalu')
)
FCVC_mapping = {'Selalu': 3, 'Sering': 2, 'Kadang-Kadang': 1, 'Tidak Pernah': 0}
FCVC_y = FCVC_mapping[FCVC_input]

NCP_input = st.selectbox(
    "Berapa Banyak Makan Utama yang Dikonsumsi Setiap Hari:",
    ('Tidak ada jawaban', 'Lebih dari 3', '3', 'antara 1&2')
)
NCP_mapping = {'Tidak ada jawaban': 3, 'Lebih dari 3': 2, '3': 1, 'antara 1&2': 0}
NCP_y = NCP_mapping[NCP_input]

SCC_input = st.selectbox(
    "Apakah Anda Memantau Asupan Kalori:",
    ('ya', 'tidak')
)
SCC_mapping = {'ya': 1, 'tidak': 0}
SCC_y = SCC_mapping[SCC_input]

Smoke_input = st.selectbox(
    "Apakah Anda Merokok:",
    ('ya', 'tidak')
)
Smoke_mapping = {'ya': 1, 'tidak': 0}
Smoke_y = Smoke_mapping[Smoke_input]

CH2O_input = st.selectbox(
    "Berapa Banyak Air yang Dikonsumsi Setiap Hari:",
    ('Lebih dari 2 Liter', 'antara 1&2 Liter', 'kurang dari 1 Liter')
)
CH2O_mapping = {'Lebih dari 2 Liter': 2, 'antara 1&2 Liter': 1, 'kurang dari 1 Liter': 0}
CH2O_y = CH2O_mapping[CH2O_input]

FHO_input = st.selectbox(
    "Apakah Anda Memiliki Anggota Keluarga yang Kelebihan Berat Badan:",
    ('ya', 'tidak')
)
FHO_mapping = {'ya': 1, 'tidak': 0}
FHO_y = FHO_mapping[FHO_input]

FAF_input = st.selectbox(
    "Seberapa Sering Anda Melakukan Aktivitas Fisik:",
    ('4/5 kali seminggu', '2/3 kali seminggu', '1/2 kali seminggu', 'tidak pernah')
)
FAF_mapping = {'4/5 kali seminggu': 3, '2/3 kali seminggu': 2, '1/2 kali seminggu': 1, 'tidak pernah': 0}
FAF_y = FAF_mapping[FAF_input]

TUE_input = st.selectbox(
    "Berapa Lama Anda Menggunakan Perangkat Elektronik:",
    ('Lebih dari 3 jam', 'antara 1 dan 3 jam', 'kurang dari 1 jam', 'tidak ada')
)
TUE_mapping = {'Lebih dari 3 jam': 3, 'antara 1 dan 3 jam': 2, 'kurang dari 1 jam': 1, 'tidak ada': 0}
TUE_y = TUE_mapping[TUE_input]

CAEC_input = st.selectbox(
    "Seberapa Sering Anda Makan di Antara Makanan:",
    ('Tidak Pernah', 'Kadang-Kadang', 'Sering', 'Selalu')
)
CAEC_mapping = {'Selalu': 3, 'Sering': 2, 'Kadang-Kadang': 1, 'Tidak Pernah': 0}
CAEC_y = CAEC_mapping[CAEC_input]

MTRANS_input = st.selectbox(
    "Jenis Transportasi Apa yang Anda Gunakan:",
    ('mobil', 'Sepeda motor', 'sepeda', 'Transportasi Umum', 'Berjalan kaki')
)
MTRANS_mapping = {'mobil': 4, 'Sepeda motor': 3, 'sepeda': 2, 'Transportasi Umum': 1, 'Berjalan kaki': 0}
MTRANS_y = MTRANS_mapping[MTRANS_input]

# Prediction code
Prediksi_Obesitas = ''
if st.button("Ayo Cek!"):
    if Age and Height and Weight:
        try:
            # Scaling numerical input features
            st.write(f"Age: {Age}, Height: {Height}, Weight: {Weight}")
            scaled_features = scaler.transform([[Age, Height, Weight]])
            st.write(f"Scaled features: {scaled_features}")
            
            # Combining all features into a single array
            features = np.array([
                scaled_features[0][0], scaled_features[0][1], scaled_features[0][2],
                Sex_y, CALC_y, FAVC_y, FCVC_y, NCP_y, SCC_y, Smoke_y, CH2O_y, FHO_y, FAF_y, TUE_y, CAEC_y, MTRANS_y
            ]).reshape(1, -1)
            st.write(f"Features: {features}")
            
            # Making prediction with Decision Tree
            Prediksi_Obesitas = diabetes_model.predict(features)
            
            # Interpreting the prediction result
            if Prediksi_Obesitas[0] == 0:
                Prediksi_Obesitas = "Insufficient Weight"
            elif Prediksi_Obesitas[0] == 1:
                Prediksi_Obesitas = "Normal Weight"
            elif Prediksi_Obesitas[0] == 2:
                Prediksi_Obesitas = "Overweight Level 1"
            elif Prediksi_Obesitas[0] == 3:
                Prediksi_Obesitas = "Overweight Level 2"
            elif Prediksi_Obesitas[0] == 4:
                Prediksi_Obesitas = "Overweight Level 3"   
            elif Prediksi_Obesitas[0] == 5:
                Prediksi_Obesitas = "Obesity Type 1" 
            elif Prediksi_Obesitas[0] == 6:
                Prediksi_Obesitas = "Obesity Type 2" 
            else:
                Prediksi_Obesitas = "tidak ditemukan jenis obesitas"
            
            st.success(Prediksi_Obesitas)
        except ValueError as e:
            st.error(f"ValueError: {e}")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
