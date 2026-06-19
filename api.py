import os
import torch
import torchaudio
import traceback
import joblib
import pandas as pd
import uvicorn
import librosa
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware


os.add_dll_directory(r"C:\ffmpeg\ffmpeg-8.1-essentials_build\ffmpeg-8.1-essentials_build\bin")
os.environ["TORCHAUDIO_USE_BACKEND_DISPATCHER"] = "1"
os.environ["TORCHAUDIO_USE_TORCHCODEC"] = "0"


from audio_processor import extract_features

app = FastAPI(title="Voice Gender Classification API")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# --- 2. PATH SETUP ---
BASE_DIR = r'C:\Users\Admin\Desktop\voice_recoginition_assighnment\voicedataset'
MODEL_DIR = os.path.join(BASE_DIR, 'model_assets')

# Load Model Assets
try:
    model = joblib.load(os.path.join(MODEL_DIR, 'gender_model.pkl'))
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    le = joblib.load(os.path.join(MODEL_DIR, 'label_encoder.pkl'))
    print(" Model assets loaded successfully.")
except Exception as e:
    print(f" Error loading model assets: {e}")


app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

@app.get("/", response_class=HTMLResponse)
def read_root():
    index_path = os.path.join(BASE_DIR, 'index.html')
    with open(index_path, "r") as f:
        return f.read()


try:
    model_vad, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
    (get_speech_timestamps, _, _, _, _) = utils 
    print("Silero VAD loaded.")
except Exception as e:
    print(f" VAD Loading Error: {e}")

def detect_speech(file_path: str) -> bool:
    try:
        y, sr = librosa.load(file_path, sr=16000)
        wav = torch.from_numpy(y)
        speech_timestamps = get_speech_timestamps(wav, model_vad, sampling_rate=16000)
        return len(speech_timestamps) > 0
    except Exception as e:
        print(f" Audio Reading Error: {e}")
        return False


@app.post("/predict-audio")
async def predict(file: UploadFile = File(...)):
    temp_path = os.path.join(BASE_DIR, f"temp_{file.filename}")
    
    try:
      
        content = await file.read()
        with open(temp_path, "wb") as buffer:
            buffer.write(content)

        print(f" Received file: {file.filename}")

       
        if not detect_speech(temp_path):
            return {"prediction": "No speech detected", "status": "error"}

        
        features_dict = extract_features(temp_path)
        
       
        model_features = {k: v for k, v in features_dict.items() if k != "is_human"}
        df = pd.DataFrame([model_features])
        
     
        cols_order = [
            'meanfreq', 'sd', 'median', 'Q25', 'Q75', 'IQR', 'skew', 'kurt',
            'sp.ent', 'sfm', 'mode', 'centroid', 'meanfun', 'minfun', 
            'maxfun', 'meandom', 'mindom', 'maxdom', 'dfrange', 'modindx'
        ]
        
    
        for col in cols_order:
            if col not in df.columns:
                df[col] = 0
        
        df = df[cols_order]

       
        scaled_data = scaler.transform(df)
        prediction_idx = model.predict(scaled_data)[0]
        gender = le.inverse_transform([prediction_idx])[0]
        
        print(f"Prediction: {gender}")

        return {
            "prediction": str(gender),
            "status": "success"
        }

    except Exception as e:
        print(f" ERROR DURING PREDICTION:\n{traceback.format_exc()}")
        return {"prediction": "Processing Error", "status": "server_error", "detail": str(e)}
    
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                print(f" Cleaned up: {temp_path}")
            except Exception as cleanup_error:
                print(f" Cleanup failed: {cleanup_error}")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)