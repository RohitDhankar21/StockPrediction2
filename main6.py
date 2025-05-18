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

# --- Transformer Model Definition ---
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
        x = x * np.sqrt(self.model_dim)  # Scale embedding
        x = x.permute(1, 0, 2)  # [seq_len, batch_size, features]
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # back to [batch_size, seq_len, features]
        return self.fc_out(x[:, -1, :]).squeeze(-1)

# --- Utility to create sequences ---
def create_sequences(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i + window])
        y.append(data[i + window])
    return np.array(X), np.array(y)

# --- Main Streamlit app ---
def main():
    st.title("Stock Price Prediction: Transformer + Linear Regression Ensemble")

    # Sidebar inputs
    stock_symbol = st.sidebar.text_input("Stock Symbol", value="AAME")
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2013-01-01"))
    end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2025-01-01"))
    window_size = st.sidebar.slider("Window Size (days)", min_value=5, max_value=30, value=10)
    epochs = st.sidebar.slider("Training Epochs", min_value=1, max_value=50, value=20)

    if start_date >= end_date:
        st.error("Error: End Date must be after Start Date.")
        return

    # Fetch data
    with st.spinner(f"Downloading data for {stock_symbol} ..."):
        data = yf.download(stock_symbol, start=start_date, end=end_date)
    if data.empty:
        st.error("No data found for this stock and date range.")
        return

    prices = data['Close'].values

    # Prepare data sequences
    X, y = create_sequences(prices, window_size)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_reshaped = X_train.reshape(-1, window_size)
    X_test_reshaped = X_test.reshape(-1, window_size)

    X_train_scaled = scaler.fit_transform(X_train_reshaped)
    X_test_scaled = scaler.transform(X_test_reshaped)

    X_train_scaled = X_train_scaled.reshape(-1, window_size, 1)
    X_test_scaled = X_test_scaled.reshape(-1, window_size, 1)

    # Convert to torch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Initialize model, optimizer, criterion
    model = TransformerModel(input_dim=1)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # Train button
    if st.button("Train Transformer Model"):
        loss_placeholder = st.empty()
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            # Update the same text line on Streamlit
            loss_placeholder.text(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

        st.success("Training completed!")

        # Train Linear Regression
        linear_model = LinearRegression()
        linear_model.fit(X_train_scaled.reshape(-1, window_size), y_train)

        # Predictions
        model.eval()
        with torch.no_grad():
            transformer_preds = model(X_test_tensor).numpy()
        linear_preds = linear_model.predict(X_test_scaled.reshape(-1, window_size))

        # Ensemble average
        ensemble_preds = (transformer_preds + linear_preds) / 2

        # Evaluation metrics
        mse = mean_squared_error(y_test, ensemble_preds)
        mae = mean_absolute_error(y_test, ensemble_preds)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_test - ensemble_preds) / y_test)) * 100

        st.write(f"**Ensemble Metrics:**")
        st.write(f"- MSE: {mse:.4f}")
        st.write(f"- MAE: {mae:.4f}")
        st.write(f"- RMSE: {rmse:.4f}")
        st.write(f"- MAPE: {mape:.2f}%")

        # Plot actual vs predicted with label
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(y_test, label="Actual Close Price")
        ax.plot(ensemble_preds, label="Ensemble Prediction")

        # Label above the plot
        ax.text(0.5, 1.02, "Ensemble Prediction vs Actual Close Price",
                transform=ax.transAxes, ha='center', fontsize=14, fontweight='bold')

        ax.set_xlabel("Test Sample Index")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)

if __name__ == "__main__":
    main()
