import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

from lstm_model import LSTMClassifier

# Parameters
SEQUENCE_LENGTH = 50
FEATURE_COLS = ['X', 'Y', 'Pressure', 'Duration', 'Orientation', 'Size']
NUM_FEATURES = len(FEATURE_COLS)
BATCH_SIZE = 32
EPOCHS = 20
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load reference session
reference = pd.read_csv("reference_session.csv").values

# Function to generate legitimate sequences (similar to reference)
def generate_legitimate(num_samples=200):
    data = []
    labels = []
    for _ in range(num_samples):
        # Add small noise to reference
        noise = np.random.normal(0, 0.1, reference.shape)
        seq = np.clip(reference + noise, 0, 1)
        data.append(seq)
        labels.append(1)  # legitimate
    return data, labels

# Function to generate fraudulent sequences (random)
def generate_fraudulent(num_samples=200):
    data = []
    labels = []
    for _ in range(num_samples):
        seq = np.random.rand(SEQUENCE_LENGTH, NUM_FEATURES)
        data.append(seq)
        labels.append(0)  # fraudulent
    return data, labels

# Generate data
legit_data, legit_labels = generate_legitimate(500)
fraud_data, fraud_labels = generate_fraudulent(500)

X = np.array(legit_data + fraud_data)
y = np.array(legit_labels + fraud_labels)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create dataset
class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = SequenceDataset(X_train, y_train)
test_dataset = SequenceDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model
model = LSTMClassifier().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {train_loss/len(train_loader):.4f}")

# Evaluation
model.eval()
y_pred = []
y_true = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(DEVICE)
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        y_pred.extend(predicted.cpu().numpy())
        y_true.extend(y_batch.numpy())

print(f"LSTM Accuracy: {accuracy_score(y_true, y_pred)}")

# Save model
torch.save(model.state_dict(), "lstm_classifier.pt")
print("Model saved as lstm_classifier.pt")