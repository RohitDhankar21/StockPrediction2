import streamlit as st
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

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
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)
        return self.fc_out(x[:, -1, :]).squeeze(-1)

# Training function
def train_model(model, train_loader, optimizer, criterion, epochs=100):
    model.train()
    training_loss_list = []
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
        training_loss_list.append(avg_loss)
    return training_loss_list

# Streamlit app main
def main():
    st.title("Stock Price Prediction: Transformer + Linear Regression Ensemble")

    selected_ticker = st.text_input("Enter Stock Ticker", "AAPL").upper()
    window_size = st.slider("Window Size (days)", 5, 30, 10)
    epochs = st.slider("Training Epochs", 5, 50, 20)
    batch_size = st.slider("Batch Size", 16, 128, 64)

    if st.button("Run Prediction"):
        data = yf.download(selected_ticker, start='2013-01-01', end='2025-01-01')
        if data.empty:
            st.error("No data found for this ticker.")
            return

        prices = data['Close'].values

        # Create sequences
        def create_sequences(data, window):
            X, y = [], []
            for i in range(len(data) - window):
                X.append(data[i:i + window])
                y.append(data[i + window])
            return np.array(X), np.array(y)

        X, y = create_sequences(prices, window_size)

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale
        scaler = StandardScaler()
        X_train_reshaped = X_train.reshape(-1, window_size)
        X_test_reshaped = X_test.reshape(-1, window_size)
        X_train_scaled = scaler.fit_transform(X_train_reshaped)
        X_test_scaled = scaler.transform(X_test_reshaped)
        X_train_scaled = X_train_scaled.reshape(-1, window_size, 1)
        X_test_scaled = X_test_scaled.reshape(-1, window_size, 1)

        # Convert to tensors
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Initialize model
        transformer_model = TransformerModel(input_dim=1)
        optimizer = optim.Adam(transformer_model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # Train model
        training_loss_list = train_model(transformer_model, train_loader, optimizer, criterion, epochs=epochs)

        # Train linear regression
        linear_model = LinearRegression().fit(X_train_scaled.reshape(-1, window_size), y_train)

        # Predictions
        transformer_model.eval()
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        with torch.no_grad():
            transformer_preds = transformer_model(X_test_tensor).numpy()

        linear_preds = linear_model.predict(X_test_scaled.reshape(-1, window_size))

        # Ensemble average
        ensemble_preds = (transformer_preds + linear_preds) / 2

        # Metrics
        mse = mean_squared_error(y_test, ensemble_preds)
        mae = mean_absolute_error(y_test, ensemble_preds)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_test - ensemble_preds) / y_test)) * 100

        st.subheader(f"{selected_ticker} Ensemble Model Performance:")
        st.write(f"MSE: {mse:.4f}")
        st.write(f"MAE: {mae:.4f}")
        st.write(f"RMSE: {rmse:.4f}")
        st.write(f"MAPE: {mape:.2f}%")

        # Plot actual vs predicted
        st.subheader("Ensemble Prediction vs Actual Closing Price")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(y_test, label="Actual Close Price")
        ax.plot(ensemble_preds, label="Ensemble Prediction")
        ax.set_title(f"{selected_ticker} - Actual vs Ensemble Predicted Close Price")
        ax.set_xlabel("Test Sample Index")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)

        # Show training loss logs inside an expander
        with st.expander("Show Training Loss Logs"):
            for epoch_idx, loss_val in enumerate(training_loss_list):
                st.write(f"Epoch {epoch_idx + 1}/{epochs}, Loss: {loss_val:.4f}")

if __name__ == "__main__":
    main()
