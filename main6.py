# stock_prediction_app.py

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

# -------------------------------
# 1. Streamlit Header
# -------------------------------
st.title("Stock Price Prediction: Transformer + ARIMA Ensemble")
st.markdown("Combining deep learning (Transformer) with classical time series (ARIMA)")

# -------------------------------
# 2. Download and prepare data
# -------------------------------
symbol = st.sidebar.text_input("Enter Stock Symbol", value="AAME")
data = yf.download(symbol, start='2013-01-01', end='2023-01-01', auto_adjust=True)

if data.empty:
    st.error("No data found. Please enter a valid stock symbol.")
    st.stop()

features = data[['Open', 'High', 'Low']]
target = data['Close']

def create_sequences(features, targets, window=10):
    X, y = [], []
    for i in range(len(features) - window):
        X.append(features.iloc[i:i + window].values.flatten())
        y.append(targets.iloc[i + window])
    return np.array(X), np.array(y)

X, y = create_sequences(features, target, window=10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -------------------------------
# 3. Define Transformer Model
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
        x = x.view(-1, 10, x.size(1) // 10)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.fc_out(x[:, -1, :])
        return x.view(-1, 1)

# -------------------------------
# 4. Train Transformer
# -------------------------------
@st.cache_data(show_spinner=False)
def train_transformer_model(X_train_scaled, y_train):
    input_dim = 64
    num_features = X_train_scaled.shape[1] // 10
    model = TransformerModel(input_dim=input_dim, num_features=num_features)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    dataset = TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model.train()
    for epoch in range(50):
        for inputs, targets in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            targets = targets.view(-1)
            loss = criterion(outputs, targets.view(-1, 1))
            loss.backward()
            optimizer.step()

    return model

with st.spinner("Training Transformer..."):
    transformer_model = train_transformer_model(X_train_scaled, y_train)

transformer_model.eval()
with torch.no_grad():
    transformer_preds = transformer_model(torch.tensor(X_test_scaled, dtype=torch.float32)).numpy().ravel()

# -------------------------------
# 5. ARIMA Forecast
# -------------------------------
with st.spinner("Fitting ARIMA model..."):
    arima_model = ARIMA(y_train, order=(2, 1, 0))
    arima_fitted = arima_model.fit()
    arima_preds = arima_fitted.forecast(steps=len(y_test)).ravel()

# -------------------------------
# 6. Ensemble Predictions
# -------------------------------
transformer_weight = 0.20
arima_weight = 0.80
ensemble_preds = (transformer_preds * transformer_weight) + (arima_preds * arima_weight)

# -------------------------------
# 7. Evaluation Metrics
# -------------------------------
mse = mean_squared_error(y_test, ensemble_preds)
mae = mean_absolute_error(y_test, ensemble_preds)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_test - ensemble_preds) / y_test)) * 100

st.subheader("Model Evaluation")
st.markdown(f"""
- **MSE:** {mse:.4f}  
- **MAE:** {mae:.4f}  
- **RMSE:** {rmse:.4f}  
- **MAPE:** {mape:.2f}%
""")

# -------------------------------
# 8. Plots
# -------------------------------
st.subheader("📈 Predictions vs Actual")

# Plot 1 - Transformer only
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(y_test, label='Actual Prices', color='blue')
ax1.plot(transformer_preds, label='Transformer Predictions', color='green')
ax1.set_title("Transformer vs Actual")
ax1.set_xlabel("Index")
ax1.set_ylabel("Price")
ax1.legend()
st.pyplot(fig1)

# Plot 2 - Ensemble
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(y_test, label='Actual Prices', color='blue')
ax2.plot(ensemble_preds, label='Ensemble (Transformer + ARIMA)', color='red')
ax2.set_title("Ensemble vs Actual")
ax2.set_xlabel("Index")
ax2.set_ylabel("Price")
ax2.legend()
st.pyplot(fig2)
