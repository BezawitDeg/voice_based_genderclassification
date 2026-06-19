import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os

def train_model(data_path, save_dir):
   
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    
    print("Pre-processing data...")
    X = df.drop('label', axis=1)
    y = df['label']
    
   
    le = LabelEncoder()
    y = le.fit_transform(y)
    
  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
 
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    print("Training RandomForest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
  
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
  
    print(f"Saving assets to {save_dir}...")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    joblib.dump(model, os.path.join(save_dir, 'gender_model.pkl'))
    joblib.dump(scaler, os.path.join(save_dir, 'scaler.pkl'))
    joblib.dump(le, os.path.join(save_dir, 'label_encoder.pkl'))
    
    print("Done!")
    return accuracy

if __name__ == "__main__":
   
    DATA_PATH = r'C:\Users\Admin\Desktop\voice_recoginition_assighnment\voicedataset\voice.csv'
    SAVE_DIR = r'C:\Users\Admin\Desktop\voice_recoginition_assighnment\voicedataset\model_assets'

   
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
        
    train_model(DATA_PATH, SAVE_DIR)