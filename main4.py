import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt



# -------------------------------
# Transformer Model Definition
# -------------------------------
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_features, num_layers=1, num_heads=1, ffn_hid_dim=128):
        super().__init__()
        self.pos_encoder = nn.Linear(num_features, input_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=ffn_hid_dim, batch_first=True),
            num_layers=num_layers)
        self.fc_out = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = x.view(-1, 10, x.size(1) // 10)  # reshape to (batch_size, seq_len, features)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.fc_out(x[:, -1, :])
        return x.view(-1, 1)

# -------------------------------
# Download stock data (cached)
# -------------------------------
@st.cache_data(show_spinner=False)
def download_data(stock_symbol):
    return yf.download(stock_symbol, start='2013-01-01', end='2023-01-01', auto_adjust=True)

# -------------------------------
# Create sequences for training
# -------------------------------
def create_sequences(features, targets, window=10):
    X, y = [], []
    for i in range(len(features) - window):
        X.append(features.iloc[i:i + window].values.flatten())
        y.append(targets.iloc[i + window])
    return np.array(X), np.array(y)

# -------------------------------
# Train Transformer model
# -------------------------------
def train_transformer_model(model, X_train, y_train, epochs=30, lr=0.001):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    for epoch in range(epochs):
        epoch_loss = 0
        for inputs, targets in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            targets = targets.view(-1, 1)  # fix shape
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
    return model

# -------------------------------
# Main pipeline to run models and ensemble
# -------------------------------
def run_model_pipeline(stock_symbol):
    # Download data
    data = download_data(stock_symbol)
    if data.empty:
        raise ValueError("No data found for symbol: " + stock_symbol)

    features = data[['Open', 'High', 'Low']]
    target = data['Close']

    # Create sequences
    X, y = create_sequences(features, target, window=10)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale data for Linear Regression & Transformer
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Linear Regression
    linear_model = LinearRegression()
    linear_model.fit(X_train_scaled, y_train)
    lr_test_preds = linear_model.predict(X_test_scaled)

    # Train Transformer
    input_dim = 64
    num_features = X_train.shape[1] // 10
    transformer_model = TransformerModel(input_dim=input_dim, num_features=num_features)
    transformer_model = train_transformer_model(transformer_model, X_train_scaled, y_train, epochs=30)
    transformer_model.eval()

    with torch.no_grad():
        transformer_test_preds = transformer_model(torch.tensor(X_test_scaled, dtype=torch.float32)).numpy().ravel()

    # Train ARIMA on training target
    arima_model = ARIMA(y_train, order=(2, 1, 0))
    arima_fitted = arima_model.fit()
    arima_preds = arima_fitted.forecast(steps=len(y_test)).ravel()

    # Ensemble weighted average of Transformer and ARIMA predictions
    transformer_weight = 0.20
    arima_weight = 0.80
    assert transformer_weight + arima_weight == 1, "Weights must sum to 1."

    ensemble_preds = (transformer_test_preds * transformer_weight) + (arima_preds * arima_weight)

    # Evaluation
    mse = mean_squared_error(y_test, ensemble_preds)
    mae = mean_absolute_error(y_test, ensemble_preds)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test - ensemble_preds) / y_test)) * 100

    return y_test, ensemble_preds, mse, mae, rmse, mape

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("Real-Time Stock Prediction with Transformer + ARIMA Ensemble")

stock_symbol = st.text_input("Enter stock symbol", value="AAME").upper()

if st.button("Run Prediction"):
    with st.spinner(f"Running prediction pipeline for {stock_symbol} ..."):
        try:
            y_test, ensemble_preds, mse, mae, rmse, mape = run_model_pipeline(stock_symbol)

            st.success(f"Prediction complete for {stock_symbol}!")
            st.write(f"**Evaluation Metrics:**")
            st.write(f"MSE: {mse:.4f}")
            st.write(f"MAE: {mae:.4f}")
            st.write(f"RMSE: {rmse:.4f}")
            st.write(f"MAPE: {mape:.2f}%")

            # Plot actual vs prediction
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(y_test, label='Actual Prices')
            ax.plot(ensemble_preds, label='Ensemble Predictions')
            ax.set_title(f"{stock_symbol} Actual vs Predicted Prices")
            ax.set_xlabel("Test Data Points")
            ax.set_ylabel("Price")
            ax.legend()
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Error: {e}")
