import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Parameters
SEQUENCE_LENGTH = 50
FEATURE_COLS = ['X', 'Y', 'Pressure', 'Duration', 'Orientation', 'Size']
NUM_FEATURES = len(FEATURE_COLS)

# Load reference session
reference = pd.read_csv("reference_session.csv").values

# Function to generate legitimate sequences (similar to reference)
def generate_legitimate(num_samples=50):
    data = []
    labels = []
    for _ in range(num_samples):
        # Add small noise to reference
        noise = np.random.normal(0, 0.1, reference.shape)
        seq = np.clip(reference + noise, 0, 1)
        data.append(seq.flatten())
        labels.append(1)  # legitimate
    return data, labels

# Function to generate fraudulent sequences (random)
def generate_fraudulent(num_samples=50):
    data = []
    labels = []
    for _ in range(num_samples):
        seq = np.random.rand(SEQUENCE_LENGTH, NUM_FEATURES)
        data.append(seq.flatten())
        labels.append(0)  # fraudulent
    return data, labels

# Function to extract statistical features (mean and std per feature)
def extract_stat_features(seq):
    seq_reshaped = seq.reshape(SEQUENCE_LENGTH, NUM_FEATURES)
    means = np.mean(seq_reshaped, axis=0)
    stds = np.std(seq_reshaped, axis=0)
    return np.concatenate([means, stds])

# Generate data
legit_data, legit_labels = generate_legitimate(100)
fraud_data, fraud_labels = generate_fraudulent(100)

# Data for SVM1 (sequence) - already flattened
X_seq = np.array(legit_data + fraud_data)
y = np.array(legit_labels + fraud_labels)

# Data for SVM2 (statistics)
legit_stat_data = [extract_stat_features(np.array(seq)) for seq in legit_data]
fraud_stat_data = [extract_stat_features(np.array(seq)) for seq in fraud_data]

X_stat = np.array(legit_stat_data + fraud_stat_data)

# Split data for both
X_seq_train, X_seq_test, y_train, y_test = train_test_split(X_seq, y, test_size=0.2, random_state=42)
X_stat_train, X_stat_test, _, _ = train_test_split(X_stat, y, test_size=0.2, random_state=42)

# Train SVM1 (sequence)
clf_seq = svm.SVC(kernel='rbf', probability=True, random_state=42)
clf_seq.fit(X_seq_train, y_train)

# Train SVM2 (statistics)
clf_stat = svm.SVC(kernel='linear', probability=True, random_state=42)
clf_stat.fit(X_stat_train, y_train)

# Evaluate
y_pred_seq = clf_seq.predict(X_seq_test)
y_pred_stat = clf_stat.predict(X_stat_test)
print(f"SVM1 (Sequence) Accuracy: {accuracy_score(y_test, y_pred_seq)}")
print(f"SVM2 (Statistics) Accuracy: {accuracy_score(y_test, y_pred_stat)}")

# Save models
joblib.dump(clf_seq, "../svm_seq_model.pkl")
joblib.dump(clf_stat, "../svm_stat_model.pkl")
print("Models saved as svm_seq_model.pkl and svm_stat_model.pkl")