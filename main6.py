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
from statsmodels.tsa.arima.model import ARIMA

st.title("Stock Price Prediction with Transformer + ARIMA Ensemble")

# --- PARAMETERS ---
stock_symbol = st.text_input("Stock Symbol", "AAPL")
start_date = st.date_input("Start Date", pd.to_datetime("2013-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2025-01-01"))
window_size = st.number_input("Window Size (sequence length)", min_value=3, max_value=50, value=10, step=1)
epochs = st.slider("Training Epochs", 1, 100, 20)
batch_size = st.number_input("Batch Size", min_value=8, max_value=256, value=64, step=8)

# --- LOAD DATA ---
@st.cache_data
def load_data(symbol, start, end):
    data = yf.download(symbol, start=start, end=end, auto_adjust=True)
    return data['Close'].values

prices = load_data(stock_symbol, start_date, end_date)
if len(prices) < window_size + 10:
    st.error("Not enough data to create sequences. Please choose different dates or stock.")
    st.stop()

# --- CREATE SEQUENCES ---
def create_sequences(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i + window])
        y.append(data[i + window])
    return np.array(X), np.array(y)

X, y = create_sequences(prices, window_size)

# --- TRAIN-TEST SPLIT ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- SCALE DATA ---
scaler_X = StandardScaler()
scaler_y = StandardScaler()

# Scale X (flatten first then reshape)
X_train_scaled = scaler_X.fit_transform(X_train.reshape(-1, window_size))
X_test_scaled = scaler_X.transform(X_test.reshape(-1, window_size))

# Scale y separately
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# Reshape back to 3D for transformer input
X_train_scaled = X_train_scaled.reshape(-1, window_size, 1)
X_test_scaled = X_test_scaled.reshape(-1, window_size, 1)

# --- TRANSFORMER MODEL ---
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
        x = x.permute(1, 0, 2)  # [seq_len, batch, features]
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # [batch, seq_len, features]
        return self.fc_out(x[:, -1, :]).squeeze(-1)

# Initialize model, criterion, optimizer
model = TransformerModel(input_dim=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Prepare data tensors and dataloader
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# --- TRAINING ---
loss_progress = []

def train_model(model, loader, optimizer, criterion, epochs):
    model.train()
    epoch_display = st.empty()  # placeholder to update the line

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
        loss_progress.append(avg_loss)
        epoch_display.text(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")  # update single line

train_button = st.button("Train Model")
if train_button:
    train_model(model, train_loader, optimizer, criterion, epochs)

    # --- PREDICTIONS ---
    model.eval()
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    with torch.no_grad():
        transformer_preds_scaled = model(X_test_tensor).numpy()

    # Inverse transform to original scale
    transformer_preds = scaler_y.inverse_transform(transformer_preds_scaled.reshape(-1, 1)).flatten()

    # ARIMA model on training target data (original scale)
    arima_model = ARIMA(y_train, order=(2, 1, 0))
    arima_fitted = arima_model.fit()

    # Forecast with ARIMA on test set length
    arima_preds = arima_fitted.forecast(steps=len(y_test))

    # --- ENSEMBLE (Weighted Average) ---
    transformer_weight = 0.20
    arima_weight = 0.80
    assert abs(transformer_weight + arima_weight - 1) < 1e-6, "Weights must sum to 1."

    ensemble_preds = (transformer_preds * transformer_weight) + (arima_preds * arima_weight)

    # --- METRICS ---
    mse = mean_squared_error(y_test, ensemble_preds)
    mae = mean_absolute_error(y_test, ensemble_preds)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test - ensemble_preds) / y_test)) * 100

    st.write(f"**Ensemble MSE:** {mse:.4f}")
    st.write(f"**Ensemble MAE:** {mae:.4f}")
    st.write(f"**Ensemble RMSE:** {rmse:.4f}")
    st.write(f"**Ensemble MAPE:** {mape:.2f}%")

    # --- PLOT RESULTS ---

    # Plot Actual Prices separately
    fig_actual, ax_actual = plt.subplots(figsize=(10, 5))
    ax_actual.plot(y_test, label='Actual Prices', color='blue')
    ax_actual.set_title(f"Actual Closing Prices for {stock_symbol}")
    ax_actual.set_xlabel("Test Sample Index")
    ax_actual.set_ylabel("Price")
    ax_actual.legend()
    st.pyplot(fig_actual)

    # Plot Ensemble Predictions separately
    fig_pred, ax_pred = plt.subplots(figsize=(10, 5))
    ax_pred.plot(ensemble_preds, label='Ensemble Predictions', color='red')
    ax_pred.set_title(f"Ensemble Predictions for {stock_symbol}\n(Transformer + ARIMA)")
    ax_pred.set_xlabel("Test Sample Index")
    ax_pred.set_ylabel("Price")
    ax_pred.legend()
    st.pyplot(fig_pred)

    # Plot training loss curve
    fig_loss, ax_loss = plt.subplots()
    ax_loss.plot(loss_progress, label="Training Loss")
    ax_loss.set_title("Training Loss Over Epochs")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    st.pyplot(fig_loss)
