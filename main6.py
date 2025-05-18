import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --------- Transformer Model ---------
class TransformerModel(nn.Module):
    def __init__(self, input_dim=1, model_dim=64, num_layers=2, num_heads=4, ffn_hid_dim=128):
        super().__init__()
        self.model_dim = model_dim
        self.input_proj = nn.Linear(input_dim, model_dim)  # from input_dim to model_dim embedding
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dim_feedforward=ffn_hid_dim, batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(model_dim, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = self.input_proj(x)  # (batch_size, seq_len, model_dim)
        x = x * np.sqrt(self.model_dim)
        x = x.permute(1, 0, 2)  # Transformer expects (seq_len, batch, model_dim)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # back to (batch, seq_len, model_dim)
        out = self.fc_out(x[:, -1, :])  # Use last token output for prediction
        return out.squeeze(-1)

# --------- Sequence creation ---------
def create_sequences(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i + window])
        y.append(data[i + window])
    return np.array(X), np.array(y)

# --------- Training Function ---------
def train_transformer(model, train_loader, optimizer, criterion, epochs=20):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        st.write(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

# --------- Main Streamlit App ---------
def main():
    st.title("Stock Prediction: Transformer + Linear Regression Ensemble")

    stock_symbol = st.text_input("Enter Stock Symbol", value="AAPL")
    if not stock_symbol:
        st.warning("Please enter a stock symbol.")
        return

    # Download data
    data_load_state = st.text("Downloading data...")
    data = yf.download(stock_symbol, start="2013-01-01", end="2025-01-01")
    data_load_state.text("")

    if data.empty:
        st.error("Failed to download data. Check symbol and try again.")
        return

    prices = data['Close'].values

    window_size = 10
    X, y = create_sequences(prices, window_size)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, window_size)).reshape(-1, window_size, 1)
    X_test_scaled = scaler.transform(X_test.reshape(-1, window_size)).reshape(-1, window_size, 1)

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # DataLoader
    batch_size = 64
    train_ds = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = TransformerModel(input_dim=1, model_dim=64, num_layers=2, num_heads=4, ffn_hid_dim=128)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    epochs = st.slider("Select epochs", 1, 100, 20)

    if st.button("Train Transformer"):
        with st.spinner("Training Transformer..."):
            train_transformer(model, train_loader, optimizer, criterion, epochs)
        st.success("Training done!")

        # Linear Regression baseline
        lr = LinearRegression().fit(X_train_scaled.reshape(-1, window_size), y_train)

        # Predict test data
        model.eval()
        with torch.no_grad():
            transformer_preds = model(X_test_tensor).numpy()
        linear_preds = lr.predict(X_test_scaled.reshape(-1, window_size))

        # Ensemble: simple average
        ensemble_preds = (transformer_preds + linear_preds) / 2

        mse = mean_squared_error(y_test, ensemble_preds)
        mae = mean_absolute_error(y_test, ensemble_preds)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_test - ensemble_preds) / y_test)) * 100
        r2 = r2_score(y_test, ensemble_preds)

        st.subheader("Evaluation Metrics")
        st.write(f"MSE: {mse:.4f}")
        st.write(f"MAE: {mae:.4f}")
        st.write(f"RMSE: {rmse:.4f}")
        st.write(f"MAPE: {mape:.2f}%")
        st.write(f"R2 Score: {r2:.4f}")

        # Plot results
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(y_test, label="Actual")
        ax.plot(ensemble_preds, label="Ensemble Prediction")
        ax.plot(transformer_preds, label="Transformer Prediction", alpha=0.6)
        ax.plot(linear_preds, label="Linear Regression Prediction", alpha=0.6)
        ax.legend()
        ax.set_title(f"Stock Price Prediction for {stock_symbol}")
        ax.set_xlabel("Test Sample Index")
        ax.set_ylabel("Price")
        st.pyplot(fig)

if __name__ == "__main__":
    main()
