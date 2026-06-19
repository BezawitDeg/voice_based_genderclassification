import joblib
import pandas as pd
import numpy as np
import os

class GenderPredictor:
    def __init__(self, model_dir = r'C:\Users\Admin\Desktop\voice_recoginition_assighnment\voicedataset\model_assets'):
        print(f"Loading model assets from {model_dir}...")
        self.model = joblib.load(os.path.join(model_dir, 'gender_model.pkl'))
        self.scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
        self.le = joblib.load(os.path.join(model_dir, 'label_encoder.pkl'))
        
    def predict(self, feature_dict):
        
        df = pd.DataFrame([feature_dict])
        
       
        X_scaled = self.scaler.transform(df)
        
     
        pred_idx = self.model.predict(X_scaled)[0]
        prob = self.model.predict_proba(X_scaled)[0]
        
 
        confidence_val = prob[pred_idx]

        
        if confidence_val < 0.75:
            gender = "This sound is not human sound"
        else:
            gender = self.le.inverse_transform([pred_idx])[0]
       
        return {
            "gender": gender,
            "confidence": f"{confidence_val * 100:.2f}%",
            "probabilities": {self.le.classes_[i]: f"{prob[i] * 100:.2f}%" for i in range(len(self.le.classes_))}
        }

if __name__ == "__main__":
    predictor = GenderPredictor()
    
    
    sample_female = {
        "meanfreq": 0.177697, "sd": 0.033104, "median": 0.180127, "Q25": 0.169114, 
        "Q75": 0.196203, "IQR": 0.027089, "skew": 2.135369, "kurt": 7.550137, 
        "sp.ent": 0.845115, "sfm": 0.222315, "mode": 0.174177, "centroid": 0.177697, 
        "meanfun": 0.182471, "minfun": 0.028470, "maxfun": 0.246154, "meandom": 0.301563, 
        "mindom": 0.007813, "maxdom": 0.726563, "dfrange": 0.718750, "modindx": 0.124239
    }
    
    result = predictor.predict(sample_female)
    print("\nSample Prediction Result:")
    print(f"Predicted Gender: {result['gender']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Details: {result['probabilities']}")