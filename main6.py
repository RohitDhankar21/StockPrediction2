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
import matplotlib.pyplot as plt

# Transformer Model Definition
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_layers=1, num_heads=1, ffn_hid_dim=128):
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
        x = x.permute(1, 0, 2)  # seq_len, batch, feature
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # batch, seq_len, feature
        return self.fc_out(x[:, -1, :]).squeeze(-1)

# Create sequences from price data
def create_sequences(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i + window])
        y.append(data[i + window])
    return np.array(X), np.array(y)

# Training function
def train_transformer_model(model, X_train, y_train, epochs=10, lr=0.001):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    epoch_losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for inputs, targets in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1), targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(loader)
        epoch_losses.append(avg_loss)
        st.write(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")
    st.session_state['epoch_losses'] = epoch_losses
    return model

def main():
    st.title("Stock Price Prediction: Transformer + Linear Regression Ensemble")

    stock_symbol = st.text_input("Enter Stock Symbol (e.g. AAPL):", value="AAME").upper()
    start_date = st.date_input("Start Date", value=pd.to_datetime("2013-01-01"))
    end_date = st.date_input("End Date", value=pd.to_datetime("2025-01-01"))
    window_size = st.slider("Window Size (days)", min_value=5, max_value=30, value=10)
    epochs = st.slider("Training Epochs", min_value=5, max_value=50, value=20)

    if st.button("Download & Train"):
        with st.spinner("Downloading data and training..."):
            data = yf.download(stock_symbol, start=start_date, end=end_date)
            if data.empty:
                st.error("No data found for this symbol and date range.")
                return
            prices = data['Close'].values

            X, y = create_sequences(prices, window_size)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Scale features
            scaler = StandardScaler()
            X_train_reshaped = X_train.reshape(-1, window_size)
            X_test_reshaped = X_test.reshape(-1, window_size)
            X_train_scaled = scaler.fit_transform(X_train_reshaped)
            X_test_scaled = scaler.transform(X_test_reshaped)
            X_train_scaled = X_train_scaled.reshape(-1, window_size, 1)
            X_test_scaled = X_test_scaled.reshape(-1, window_size, 1)

            # Initialize Transformer model
            transformer_model = TransformerModel(input_dim=1)

            # Train Transformer
            trained_model = train_transformer_model(transformer_model, X_train_scaled, y_train, epochs=epochs)

            # Train Linear Regression
            linear_model = LinearRegression().fit(X_train_scaled.reshape(-1, window_size), y_train)

            # Predict on test data
            trained_model.eval()
            with torch.no_grad():
                X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
                transformer_preds = trained_model(X_test_tensor).numpy()

            linear_preds = linear_model.predict(X_test_scaled.reshape(-1, window_size))

            # Ensemble predictions
            ensemble_preds = (transformer_preds + linear_preds) / 2

            # Metrics
            mse = mean_squared_error(y_test, ensemble_preds)
            mae = mean_absolute_error(y_test, ensemble_preds)
            rmse = np.sqrt(mse)
            mape = np.mean(np.abs((y_test - ensemble_preds) / y_test)) * 100

            st.success("Training and prediction completed!")
            st.write(f"MSE: {mse:.4f}")
            st.write(f"MAE: {mae:.4f}")
            st.write(f"RMSE: {rmse:.4f}")
            st.write(f"MAPE: {mape:.2f}%")

            # Plot results with labels
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(y_test, label="Actual Close Price", linewidth=2)
            ax.plot(ensemble_preds, label="Ensemble Prediction (Transformer + LR)", linewidth=2)
            ax.set_title(f"{stock_symbol} - Actual vs Ensemble Predicted Close Price")
            ax.set_xlabel("Test Sample Index")
            ax.set_ylabel("Price")
            ax.legend(loc='upper left', fontsize=12)
            st.pyplot(fig)

            # Show epoch losses in expandable section
            if 'epoch_losses' in st.session_state:
                with st.expander("Show Transformer Training Loss Logs"):
                    for i, loss in enumerate(st.session_state['epoch_losses'], 1):
                        st.write(f"Epoch {i}, Loss: {loss:.6f}")

if __name__ == "__main__":
    main()
