from fastapi import FastAPI, Request, Form, Query, HTTPException
from pydantic import BaseModel
import streamlit as st
import pickle
import pandas as pd
import uvicorn

# Load ML model
def open_model(model_path):
    """Helper function for loading model"""
    with open(model_path, "rb") as model_file:
        model = pickle.load(model_file)
    return model

model_last = open_model("https://github.com/54RaSuWi/dataset/blob/main/model.pkl") 

# Create application
app = FastAPI()

class HeartDisease(BaseModel):
    generasi: str = Query(..., description="umur 12-27 (gen Z), umur 28-43 (millenials/gen Y), umur 44-59 (gen X), umur 60-78 (baby boomer)")
    gender: str = Query(..., description="Jenis kelamin anda (0 = wanita, 1 = pria)")
    cp: int = Query(..., description="Jenis penyakit dada yang diderita (0 = typical angina, Value 1 = atypical angina, Value 2 = non-anginal pain, Value 3 = asymptomatic)")
    trestbps: int = Query(..., description="Tekanan darah (0 = Normal, 1 = Hipertensi 1, 2 = Hipertensi 2, 3 = Hipertensi 3)")
    chol: int = Query(..., description="kadar kolesterol (0 = normal, 1 = berisiko, 2 = tinggi)")
    fbs: float = Query(..., description="Kadar gula (0 = kurang dari sama 120 mg/dl, 1 = lebih dari 120 mg/dl)")
    restecg: int = Query(..., description="Kondisi EKG (0 = normal, 1 = hipertrofi vertikel kiri, 2 = gelombang kelainan ST-T)")
    thalach: float = Query(..., description="Detak jantung maksimum")
    exang: int = Query(..., description="Apakah dada nyeri saat berolahraga? (0 = tidak nyeri, 1 = nyeri)")
    oldpeak: float = Query(..., description="Perubahan depresi ST saat sistolik dibanding diastolik")
    slope: int = Query(..., description="Kemiringan depresi ST selepas berolahraga (0 = upsloping, 1 = flat, 2 = downsloping)")
    ca: int = Query(..., description="Banyaknya pembuluh darah yang terdeteksi oleh fluoroskopi (0,1,2,3,4)")
    thal: int = Query(..., description="Jenis thalasemia (0 = normal, 1 = fixed defect (permanen), 2 & 3 : reversal defect (sementara))")

# Define FastAPI endpoint URL
base_url = "http://localhost:8000"


@app.post('/predict')
def predict():
    st.title("Probabilitas Pengidap Penyakit Jantung")
    
    # User input fields
    generasi = st.selectbox("Generasi", ["gen Z", "millenials", "gen X",  "baby boomer"])
    gender = st.selectbox("Jenis kelamin", [0, 1])
    cp = st.selectbox("Jenis nyeri dada yang diderita", [0, 1, 2])
    trestbps = st.selectbox("Tekanan darah pasien saat istirahat", [0, 1, 2, 3])
    chol = st.selectbox("Kadar kolesterol", [0, 1, 2])
    fbs = st.selectbox("Kadar gula darah", [0, 1])
    restecg = st.selectbox("Kondisi EKG", [0, 1, 2])
    thalach = st.number_input("Detak jantung maksimum", min_value=0.0, value=100.0, max_value=500.0)
    exang = st.selectbox("Nyeri dada saat olahraga", [0, 1])
    oldpeak = st.number_input("Perubahan depresi ST saat olahraga relatif saat istirahat", value=0.0)
    slope = st.selectbox("Kemiringan ST setelah olaraga", [0, 1, 2])
    ca = st.selectbox("Jumlah pembuluh darah besar yang terdeteksi", [0, 1, 2, 4])
    thal = st.selectbox("Jenis thalasemia", [0, 1, 2, 3])


    if st.button("Predict"):
        data = HeartDisease(
            generasi=generasi,
            gender=gender,
            cp=cp,
            trestbps=trestbps,
            chol=chol,
            fbs=fbs,
            restecg=restecg,
            thalach=thalach,
            exang=exang,
            oldpeak=oldpeak,
            slope=slope,
            ca=ca,
            thal=thal
        )

        prediction = model_last.predict(data)
        output = prediction.json()
        if output == 1:
            result = 'Pasien mempunyai risiko terkena penyakit jantung'                    
        else:
            result = 'Pasien tidak mempunyai risiko terkena penyakit jantung'
    
if __name__ == "__main__":
    uvicorn.run(app)
