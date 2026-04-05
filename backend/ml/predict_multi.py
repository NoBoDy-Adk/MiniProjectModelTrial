import pandas as pd
import numpy as np
import joblib
import torch
from lstm_model import LSTMClassifier

# Load SVM models
clf_seq = joblib.load("../svm_seq_model.pkl")
clf_stat = joblib.load("../svm_stat_model.pkl")

# Load LSTM classifier
lstm_model = LSTMClassifier()
lstm_model.load_state_dict(torch.load("lstm_classifier.pt", map_location=torch.device("cpu")))
lstm_model.eval()

feature_cols = ['X', 'Y', 'Pressure', 'Duration', 'Orientation', 'Size']

def pad_sequence(df):
    arr = df[feature_cols].values
    if len(arr) >= 50:
        seq = arr[:50]
    else:
        padding = np.zeros((50 - len(arr), len(feature_cols)))
        seq = np.vstack([arr, padding])
    return seq

def extract_stat_features(seq):
    means = np.mean(seq, axis=0)
    stds = np.std(seq, axis=0)
    return np.concatenate([means, stds])

# Load test data
df = pd.read_csv("temp_input.csv")
test_seq = pad_sequence(df)

# SVM1 score (sequence)
test_seq_flat = test_seq.flatten()
svm1_score = clf_seq.predict_proba([test_seq_flat])[0][1]

# SVM2 score (statistics)
stat_features = extract_stat_features(test_seq)
svm2_score = clf_stat.predict_proba([stat_features])[0][1]

# LSTM score (classifier)
test_tensor = torch.tensor(test_seq, dtype=torch.float32).unsqueeze(0)
with torch.no_grad():
    outputs = lstm_model(test_tensor)
    lstm_score = outputs[0][1].item()  # Probability of class 1 (legitimate)

print(f"{svm1_score},{svm2_score},{lstm_score}")