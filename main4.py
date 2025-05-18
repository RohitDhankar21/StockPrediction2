import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

st.title("Real-Time Stock Price Prediction with Transformer + ARIMA + Linear Regression Ensemble")

# ------------- Select stock symbol ----------------
stock_symbol = st.text_input("Enter stock symbol (e.g. AAPL, MSFT, TSLA)", "AAPL")

# ------------- Download Data ----------------------
@st.cache_data(ttl=3600)
def load_data(symbol):
    # Download last 5 years daily data
    data = yf.download(symbol, period="5y", auto_adjust=True)
    return data

data = load_data(stock_symbol)

if data.empty:
    st.error("Failed to fetch data. Please check the symbol and try again.")
    st.stop()

st.write(f"Showing data for {stock_symbol}")
st.line_chart(data['Close'])

# ------------- Prepare sequences for models -------------
def create_sequences(features, targets, window=10):
    X, y = [], []
    for i in range(len(features) - window):
        X.append(features.iloc[i:i + window].values.flatten())
        y.append(targets.iloc[i + window])
    return np.array(X), np.array(y)

features = data[['Open', 'High', 'Low']]
target = data['Close']
window_size = 10

X, y = create_sequences(features, target, window=window_size)

# Split train/test: last 20% for testing
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------- Define Transformer Model -------------
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_features, num_layers=1, num_heads=2, ffn_hid_dim=128):
        super().__init__()
        self.pos_encoder = nn.Linear(num_features, input_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=ffn_hid_dim, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(input_dim, 1)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, window_size, -1)  # reshape to (batch_size, seq_len, features)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.fc_out(x[:, -1, :])  # take last time step
        return x.view(-1)

# ------------- Train Transformer -------------
def train_transformer_model(model, X_train, y_train, epochs=20, lr=0.001):
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
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if epoch % 5 == 0:
            st.write(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(loader):.6f}")
    return model

# ------------- Train models --------------------
with st.spinner('Training models... This may take a minute.'):

    # Linear Regression
    linear_model = LinearRegression()
    linear_model.fit(X_train_scaled, y_train)

    # Transformer
    input_dim = 64
    num_features = X_train.shape[1] // window_size
    transformer_model = TransformerModel(input_dim=input_dim, num_features=num_features)
    transformer_model = train_transformer_model(transformer_model, X_train_scaled, y_train)

    transformer_model.eval()
    with torch.no_grad():
        transformer_test_preds = transformer_model(torch.tensor(X_test_scaled, dtype=torch.float32)).numpy()

    # ARIMA on training target only
    try:
        arima_model = ARIMA(y_train, order=(2, 1, 0))
        arima_fitted = arima_model.fit()
        arima_preds = arima_fitted.forecast(steps=len(y_test)).values
    except Exception as e:
        st.warning(f"ARIMA model failed: {e}")
        arima_preds = np.zeros_like(y_test)

# ------------- Ensemble prediction -------------
transformer_weight = 0.2
arima_weight = 0.8

ensemble_preds = transformer_test_preds * transformer_weight + arima_preds * arima_weight

# ------------- Evaluation -------------
mse = mean_squared_error(y_test, ensemble_preds)
mae = mean_absolute_error(y_test, ensemble_preds)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_test - ensemble_preds) / y_test)) * 100

st.write(f"### Ensemble Model Performance on Test Set")
st.write(f"MSE: {mse:.4f}")
st.write(f"MAE: {mae:.4f}")
st.write(f"RMSE: {rmse:.4f}")
st.write(f"MAPE: {mape:.2f}%")

# ------------- Plot predictions -------------
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(y_test, label='Actual Close Price')
ax.plot(ensemble_preds, label='Ensemble Prediction')
ax.set_title(f"{stock_symbol} - Actual vs Ensemble Prediction")
ax.set_xlabel("Test Data Points")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)
