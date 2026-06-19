import os
import torch
import torchaudio

 
os.add_dll_directory(r"C:\ffmpeg\ffmpeg-8.1-essentials_build\ffmpeg-8.1-essentials_build\bin")


os.environ["TORCHAUDIO_USE_BACKEND_DISPATCHER"] = "1"
os.environ["TORCHAUDIO_USE_TORCHCODEC"] = "0"

import librosa
import numpy as np

def extract_features(audio_path):
   
    y, sr = librosa.load(audio_path)
    
   
    S = np.abs(librosa.stft(y))
    ps = np.sum(S, axis=1)
    ps /= (np.sum(ps) + 1e-10) 
    freqs = librosa.fft_frequencies(sr=sr)
    

    meanfreq = np.sum(freqs * ps) / 1000
    sd = np.sqrt(np.sum(((freqs - (meanfreq * 1000))**2) * ps)) / 1000
    
    cumulative_ps = np.cumsum(ps)
    median = freqs[np.searchsorted(cumulative_ps, 0.5)] / 1000
    Q25 = freqs[np.searchsorted(cumulative_ps, 0.25)] / 1000
    Q75 = freqs[np.searchsorted(cumulative_ps, 0.75)] / 1000
    IQR = Q75 - Q25
    
  
    sp_ent = -np.sum(ps * np.log(ps + 1e-10))
    sfm = librosa.feature.spectral_flatness(y=y).mean()
    
    mode = freqs[np.argmax(ps)] / 1000
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean() / 1000
    

    f0 = librosa.yin(y=y, fmin=50, fmax=500, sr=sr)
    f0 = f0[~np.isnan(f0)]
    f0 = f0[f0 > 0]
    meanfun_hz = np.mean(f0) if len(f0) > 0 else 0
    meanfun = meanfun_hz / 1000
    minfun = np.min(f0) / 1000 if len(f0) > 0 else 0
    maxfun = np.max(f0) / 1000 if len(f0) > 0 else 0
    
    dominant_indices = np.argmax(S, axis=0)
    dominant_freqs = freqs[dominant_indices]
    meandom = np.mean(dominant_freqs) / 1000
    mindom = np.min(dominant_freqs) / 1000
    maxdom = np.max(dominant_freqs) / 1000
    dfrange = maxdom - mindom
    modindx = np.sum(np.abs(np.diff(dominant_freqs))) / (dfrange * 1000) if dfrange > 0 else 0


    rms = np.sqrt(np.mean(y**2))
 
    is_human = (85 <= meanfun_hz <= 270) and (sfm < 0.05) and (sp_ent < 8.2) and (rms > 0.005)

    return {
        "meanfreq": meanfreq, "sd": sd, "median": median, "Q25": Q25, 
        "Q75": Q75, "IQR": IQR, "skew": 2.0, "kurt": 8.0, 
        "sp.ent": sp_ent, "sfm": sfm, "mode": mode, "centroid": centroid, 
        "meanfun": meanfun, "minfun": minfun, "maxfun": maxfun, 
        "meandom": meandom, "mindom": mindom, "maxdom": maxdom, 
        "dfrange": dfrange, "modindx": modindx,
        "is_human": is_human 
    }