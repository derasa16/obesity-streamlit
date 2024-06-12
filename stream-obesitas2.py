import streamlit as st
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

st.write("Starting application...")

try:
    st.write("All modules imported successfully!")
    
    # Membaca model
    diabetes_model = pickle.load(open('obesitas_model.sav', 'rb'))

    # Membaca scaler
    scaler = pickle.load(open('StandardScaler.pkl', 'rb'))
    
    # Membaca label encoder
    label_encoders = pickle.load(open('le .pkl', 'rb'))
    
    st.write("Model, scaler, and label encoder loaded successfully!")
    st.write("Label encoders keys:", list(label_encoders.keys()))
    
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

# Dropdown input fields
Sex_input = st.selectbox("Pilih jenis kelamin:", ('Laki-Laki', 'Perempuan'))
CALC_input = st.selectbox("Seberapa Sering Mengkonsumsi Alkohol:", ('Tidak Pernah', 'Kadang-Kadang', 'Sering', 'Selalu'))
FAVC_input = st.selectbox("Apakah Anda Sering Mengkonsumsi Makanan Tinggi Kalori:", ('ya', 'tidak'))
SCC_input = st.selectbox("Apakah Anda Memantau Asupan Kalori:", ('ya', 'tidak'))
Smoke_input = st.selectbox("Apakah Anda Merokok:", ('ya', 'tidak'))
FHO_input = st.selectbox("Apakah Anda Memiliki Anggota Keluarga yang Kelebihan Berat Badan:", ('ya', 'tidak'))
CAEC_input = st.selectbox("Seberapa Sering Anda Makan di Antara Makanan:", ('Tidak Pernah', 'Kadang-Kadang', 'Sering', 'Selalu'))
MTRANS_input = st.selectbox("Jenis Transportasi Apa yang Anda Gunakan:", ('mobil', 'Sepeda motor', 'sepeda', 'Transportasi Umum', 'Berjalan kaki'))

# Check if label encoders are loaded correctly and if the keys match
required_keys = ['Sex', 'CALC', 'FAVC', 'SCC', 'Smoke', 'FHO', 'CAEC', 'MTRANS', 'NObeyesdad']
if all(key in label_encoders for key in required_keys):
    try:
        # Encoding categorical input fields using the loaded label encoders
        Sex_y = label_encoders['Sex'].transform([Sex_input])[0]
        CALC_y = label_encoders['CALC'].transform([CALC_input])[0]
        FAVC_y = label_encoders['FAVC'].transform([FAVC_input])[0]
        SCC_y = label_encoders['SCC'].transform([SCC_input])[0]
        Smoke_y = label_encoders['Smoke'].transform([Smoke_input])[0]
        FHO_y = label_encoders['FHO'].transform([FHO_input])[0]
        CAEC_y = label_encoders['CAEC'].transform([CAEC_input])[0]
        MTRANS_y = label_encoders['MTRANS'].transform([MTRANS_input])[0]
    except KeyError as e:
        st.error(f"KeyError: {e}")
    except Exception as e:
        st.error(f"An error occurred while encoding: {e}")
else:
    st.error("Label encoders are not loaded correctly or keys are missing.")

# Other input fields
FCVC_input = st.selectbox("Seberapa Sering Mengkonsumsi Sayuran:", ('Tidak Pernah', 'Kadang-Kadang', 'Sering', 'Selalu'))
FCVC_mapping = {'Selalu': 3, 'Sering': 2, 'Kadang-Kadang': 1, 'Tidak Pernah': 0}
FCVC_y = FCVC_mapping[FCVC_input]

NCP_input = st.selectbox("Berapa Banyak Makan Utama yang Dikonsumsi Setiap Hari:", ('Tidak ada jawaban', 'Lebih dari 3', '3', 'antara 1&2'))
NCP_mapping = {'Tidak ada jawaban': 3, 'Lebih dari 3': 2, '3': 1, 'antara 1&2': 0}
NCP_y = NCP_mapping[NCP_input]

CH2O_input = st.selectbox("Berapa Banyak Air yang Dikonsumsi Setiap Hari:", ('Lebih dari 2 Liter', 'antara 1&2 Liter', 'kurang dari 1 Liter'))
CH2O_mapping = {'Lebih dari 2 Liter': 2, 'antara 1&2 Liter': 1, 'kurang dari 1 Liter': 0}
CH2O_y = CH2O_mapping[CH2O_input]

FAF_input = st.selectbox("Seberapa Sering Anda Melakukan Aktivitas Fisik:", ('4/5 kali seminggu', '2/3 kali seminggu', '1/2 kali seminggu', 'tidak pernah'))
FAF_mapping = {'4/5 kali seminggu': 3, '2/3 kali seminggu': 2, '1/2 kali seminggu': 1, 'tidak pernah': 0}
FAF_y = FAF_mapping[FAF_input]

TUE_input = st.selectbox("Berapa Lama Anda Menggunakan Perangkat Elektronik:", ('Lebih dari 3 jam', 'antara 1 dan 3 jam', 'kurang dari 1 jam', 'tidak ada'))
TUE_mapping = {'Lebih dari 3 jam': 3, 'antara 1 dan 3 jam': 2, 'kurang dari 1 jam': 1, 'tidak ada': 0}
TUE_y = TUE_mapping[TUE_input]

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
            Prediksi_Obesitas_encoded = diabetes_model.predict(features)
            
            # Decode the prediction result
            Prediksi_Obesitas = label_encoders['NObeyesdad'].inverse_transform(Prediksi_Obesitas_encoded)[0]
            
            st.success(Prediksi_Obesitas)
        except ValueError as e:
            st.error(f"ValueError: {e}")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
