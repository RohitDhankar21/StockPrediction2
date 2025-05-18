import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# --- Transformer Model Definition ---
class SimpleTransformer(nn.Module):
    def __init__(self, feature_size=1, num_layers=1, num_heads=1, d_model=64):
        super(SimpleTransformer, self).__init__()
        self.input_proj = nn.Linear(feature_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, src):
        # src shape: (seq_len, batch, feature_size)
        src = self.input_proj(src)
        output = self.transformer(src)
        output = self.decoder(output)
        return output.squeeze(-1)  # output shape: (seq_len, batch)

# --- Prepare data ---
def prepare_data(df, seq_len=30):
    data = df['Close'].values
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    X = np.array(X)
    y = np.array(y)
    return X, y

# --- Train transformer model ---
def train_transformer(X_train, y_train, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleTransformer()
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                            torch.tensor(y_train, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            xb = xb.permute(1, 0).unsqueeze(-1).to(device)  # (seq_len, batch, feature_size=1)
            yb = yb.to(device).squeeze()                    # (batch)
            optimizer.zero_grad()
            outputs = model(xb)                             # (seq_len, batch)
            outputs = outputs[-1]                           # last time step prediction, shape (batch)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
    return model

# --- Predict with transformer ---
def predict_transformer(model, X):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32).permute(1, 0).unsqueeze(-1).to(device)
    with torch.no_grad():
        outputs = model(X_t)
    preds = outputs[-1].cpu().numpy()  # shape (batch,)
    return preds

# --- ARIMA model prediction ---
def predict_arima(series, steps):
    model = ARIMA(series, order=(5,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast

# --- Streamlit app ---
def main():
    st.title("Stock Price Prediction - Transformer + ARIMA Ensemble")

    ticker = st.text_input("Enter Stock Ticker", "AAPL")
    data_load_state = st.text("Loading data...")
    df = yf.download(ticker, period="1y")
    data_load_state.text("Loading data...done!")

    st.line_chart(df['Close'])

    # Prepare dataset
    seq_len = 30
    X, y = prepare_data(df, seq_len)
    train_size = int(len(X) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    if st.button("Train Transformer Model"):
        with st.spinner("Training transformer..."):
            model = train_transformer(X_train, y_train, epochs=5)
        st.success("Transformer trained!")

        # Predictions
        transformer_preds = predict_transformer(model, X_test)

        # ARIMA prediction on full closing series for test length
        arima_preds = predict_arima(df['Close'].values, steps=len(y_test))

        # Ensemble (weighted average)
        transformer_weight = 0.7
        ensemble_preds = transformer_preds * transformer_weight + arima_preds * (1 - transformer_weight)

        # Flatten to 1D arrays
        y_true_flat = y_test.flatten()
        ensemble_preds_flat = ensemble_preds.flatten()

        # Metrics
        mse = mean_squared_error(y_true_flat, ensemble_preds_flat)
        mae = mean_absolute_error(y_true_flat, ensemble_preds_flat)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_true_flat - ensemble_preds_flat) / y_true_flat)) * 100

        st.write(f"MSE: {mse:.4f}")
        st.write(f"MAE: {mae:.4f}")
        st.write(f"RMSE: {rmse:.4f}")
        st.write(f"MAPE: {mape:.2f}%")

        # Plot predictions
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12,6))
        plt.plot(y_true_flat, label='True')
        plt.plot(ensemble_preds_flat, label='Ensemble Prediction')
        plt.legend()
        st.pyplot(plt)

if __name__ == "__main__":
    main()
