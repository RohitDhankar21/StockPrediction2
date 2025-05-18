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

st.title("Stock Price Prediction with Transformer + Linear Regression + ARIMA Ensemble")

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
    data = yf.download(symbol, start=start, end=end)
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

X_train_scaled = scaler_X.fit_transform(X_train.reshape(-1, window_size))
X_test_scaled = scaler_X.transform(X_test.reshape(-1, window_size))
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

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
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)
        return self.fc_out(x[:, -1, :]).squeeze(-1)

model = TransformerModel(input_dim=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

loss_progress = []

def train_model(model, loader, optimizer, criterion, epochs):
    model.train()
    epoch_display = st.empty()

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
        epoch_display.text(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

train_button = st.button("Train Model")
if train_button:
    train_model(model, train_loader, optimizer, criterion, epochs)

    model.eval()
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    with torch.no_grad():
        transformer_preds_scaled = model(X_test_tensor).numpy()

    transformer_preds = scaler_y.inverse_transform(transformer_preds_scaled.reshape(-1, 1)).flatten()

    linear_model = LinearRegression()
    linear_model.fit(X_train_scaled.reshape(-1, window_size), y_train_scaled)
    linear_preds_scaled = linear_model.predict(X_test_scaled.reshape(-1, window_size))
    linear_preds = scaler_y.inverse_transform(linear_preds_scaled.reshape(-1, 1)).flatten()

    # --- ARIMA MODEL ---
    arima_model = ARIMA(y_train, order=(2, 1, 0))
    arima_fitted = arima_model.fit()
    arima_preds = arima_fitted.forecast(steps=len(y_test)).ravel()

    # --- ENSEMBLE PREDICTIONS ---
    transformer_weight = 0.2
    arima_weight = 0.8
    transformer_preds = transformer_preds.ravel()
    arima_preds = arima_preds.ravel()
    ensemble_preds = (transformer_weight * transformer_preds) + (arima_weight * arima_preds)

    mse = mean_squared_error(y_test, ensemble_preds)
    mae = mean_absolute_error(y_test, ensemble_preds)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test - ensemble_preds) / y_test)) * 100

    st.write(f"**Ensemble MSE:** {mse:.4f}")
    st.write(f"**Ensemble MAE:** {mae:.4f}")
    st.write(f"**Ensemble RMSE:** {rmse:.4f}")
    st.write(f"**Ensemble MAPE:** {mape:.2f}%")

    ensemble_name = "Transformer + ARIMA (Weighted Ensemble)"

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(y_test, label='Actual Prices', color='blue')
    ax.plot(ensemble_preds, label='Ensemble Predictions', color='red')
    ax.set_title(f"Actual vs Predicted Closing Prices for {stock_symbol}\n({ensemble_name})")
    ax.set_xlabel("Test Sample Index")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)

    fig2, ax2 = plt.subplots()
    ax2.plot(loss_progress, label="Training Loss")
    ax2.set_title("Training Loss Over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    st.pyplot(fig2)
