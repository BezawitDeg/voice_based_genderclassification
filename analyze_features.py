import joblib
import pandas as pd
import numpy as np
import os

def analyze_features(model_path, data_path):
    print("Loading model and data for feature analysis...")
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)
    X = df.drop('label', axis=1)
    
    importances = model.feature_importances_
    feature_names = X.columns
    

    indices = np.argsort(importances)[::-1]
    
    print("\nFeature Importance Ranking:")
    for f in range(X.shape[1]):
        print(f"{f + 1}. {feature_names[indices[f]]}: {importances[indices[f]]:.4f}")

if __name__ == "__main__":
    
    MODEL_PATH = r'C:\Users\Admin\Desktop\voice_recoginition_assighnment\voicedataset\model_assets\gender_model.pkl'
    DATA_PATH = r'C:\Users\Admin\Desktop\voice_recoginition_assighnment\voicedataset\voice.csv'

    if os.path.exists(MODEL_PATH):
        analyze_features(MODEL_PATH, DATA_PATH)
    else:
        print(f"Model not found at: {MODEL_PATH}")
        print("Please check if the folder name or file name is spelled correctly.")
