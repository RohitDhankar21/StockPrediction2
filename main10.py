import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# --- Streamlit title ---
st.title("Stock Prediction with Transformer + Linear Regression (Stacked Ensemble)")

# --- Inputs ---
stock_symbol = st.text_input("Stock Symbol", "AAPL")
start_date = st.date_input("Start Date", pd.to_datetime("2013-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2025-01-01"))
window_size = st.number_input("Window Size (sequence length)", min_value=3, max_value=50, value=10, step=1)
epochs = st.slider("Training Epochs", 1, 100, 20)
batch_size = st.number_input("Batch Size", min_value=8, max_value=256, value=64, step=8)

# --- Data loading ---
@st.cache_data
def load_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end)
    return df[['Close']]

df = load_data(stock_symbol, start_date, end_date)
if df.isna().sum().sum() > 0 or len(df) < window_size + 20:
    st.error("Not enough or invalid data. Choose a different stock/date range.")
    st.stop()

prices = df['Close'].values

# Create sequences for training
def create_sequences(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i + window])
        y.append(data[i + window])
    return np.array(X), np.array(y)

X, y = create_sequences(prices, window_size)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape for transformer input: [batch, seq_len, features=1]
X_train_scaled = X_train_scaled.reshape(-1, window_size, 1)
X_test_scaled = X_test_scaled.reshape(-1, window_size, 1)

# Transformer Model (like your second code)
class TransformerModel(nn.Module):
    def __init__(self, input_dim=1, num_layers=1, num_heads=1, ffn_hid_dim=128):
        super().__init__()
        self.model_dim = input_dim
        self.pos_encoder = nn.Linear(input_dim, self.model_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.model_dim, nhead=num_heads, dim_feedforward=ffn_hid_dim),
            num_layers=num_layers)
        self.fc_out = nn.Linear(self.model_dim, 1)

    def forward(self, x):
        x = self.pos_encoder(x)
        x = x * np.sqrt(self.model_dim)
        x = x.permute(1, 0, 2)  # seq_len, batch, features
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # batch, seq_len, features
        return self.fc_out(x[:, -1, :]).squeeze(-1)

# Initialize model, optimizer, criterion
transformer = TransformerModel()
optimizer = optim.Adam(transformer.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Convert to torch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

# Dataloader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Train function with progress bar
def train_transformer(model, loader, optimizer, criterion, epochs):
    model.train()
    progress_bar = st.progress(0)
    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        progress_bar.progress((epoch + 1) / epochs)
        st.write(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")
    st.success("Training Complete!")

train_button = st.button("Train Ensemble")

if train_button:
    train_transformer(transformer, train_loader, optimizer, criterion, epochs)

    transformer.eval()
    with torch.no_grad():
        transformer_preds = transformer(X_test_tensor).numpy()

    # Linear regression
    linear_model = LinearRegression().fit(X_train_scaled.reshape(-1, window_size), y_train)
    linear_preds = linear_model.predict(X_test_scaled.reshape(-1, window_size))

    # Stacked ensemble: average predictions
    ensemble_preds = (transformer_preds + linear_preds) / 2

    # Metrics
    mse = mean_squared_error(y_test, ensemble_preds)
    mae = mean_absolute_error(y_test, ensemble_preds)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test - ensemble_preds) / y_test)) * 100

    st.subheader("📊 Performance Metrics")
    st.write(f"**MSE:** {mse:.6f}")
    st.write(f"**MAE:** {mae:.6f}")
    st.write(f"**RMSE:** {rmse:.6f}")
    st.write(f"**MAPE:** {mape:.2f}%")

    # Pretty plot without grid lines or stripes
    st.subheader("📈 Actual vs Predicted Closing Prices")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(y_test, label="Actual", color='royalblue', linewidth=2)
    ax.plot(ensemble_preds, label="Ensemble Prediction", color='orangered', linestyle='--', linewidth=2)
    ax.set_title(f"{stock_symbol} - Actual vs Predicted (Ensemble)")
    ax.legend()
    ax.grid(False)  # Remove grid lines
    st.pyplot(fig)
