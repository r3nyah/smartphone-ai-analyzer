import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import joblib 

# Configuration
DATA_PATH = "data/processed/training_data.csv"
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 500

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')

class PerformancePredictor(nn.Module):
    def __init__(self, input_dim):
        super(PerformancePredictor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.model(x)

def train():
    device = get_device()
    if not os.path.exists(DATA_PATH):
        print("Error: Dataset not found.")
        return

    df = pd.read_csv(DATA_PATH)
    df = df[(df['clean_price'] > 0) & (df['clean_ram'] > 0)]
    
    X = df[['clean_ram', 'storage', 'clean_price']].values
    y = df['antutu_score'].values
    
    # Scale Data
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_x.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1))

    # --- SAVE SCALERS (CRITICAL FOR APP) ---
    joblib.dump(scaler_x, 'scaler_x.pkl')
    joblib.dump(scaler_y, 'scaler_y.pkl')
    print("[+] Scalers saved successfully!")

    # Split & Convert
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    
    dataset = TensorDataset(X_train_tensor, y_train_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Init Model
    model = PerformancePredictor(input_dim=X.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train
    print("[+] Training started...")
    model.train()
    for epoch in range(EPOCHS):
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
    # Save Model
    torch.save(model.state_dict(), "model_antutu.pth")
    print("[+] Model saved as 'model_antutu.pth'")

if __name__ == "__main__":
    train()