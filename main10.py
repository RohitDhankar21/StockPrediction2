# Streamlit app with stacked Transformer + Linear Regression, future prediction, and save/load support

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
import datetime
import joblib
import os

st.set_page_config(page_title="Stacked Stock Prediction", layout="wide")

st.title("Stock Price Prediction with Stacked Transformer + Linear Regression")

# Sidebar
st.sidebar.header("Settings")

symbol = st.sidebar.text_input("Stock Symbol", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2013-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2023-01-01"))
window_size = st.sidebar.number_input("Window Size", min_value=3, max_value=50, value=10, step=1)
epochs = st.sidebar.slider("Training Epochs", 1, 100, 20)
batch_size = st.sidebar.number_input("Batch Size", 8, 256, 64, step=8)
forecast_days = st.sidebar.slider("Forecast Days", 1, 60, 15)

model_dir = "models"
os.makedirs(model_dir, exist_ok=True)

@st.cache_data

def load_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end)
    return df[['Open', 'High', 'Low', 'Close']]

# Model definition
class TransformerModel(nn.Module):
    def __init__(self, input_dim, feature_count):
        super().__init__()
        self.linear = nn.Linear(feature_count, input_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=1, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.output = nn.Linear(input_dim, 1)

    def forward(self, x):
        B = x.size(0)
        seq_len = 10
        feat = x.size(1) // seq_len
        x = x.view(B, seq_len, feat)
        x = self.linear(x)
        x = self.transformer(x)
        return self.output(x[:, -1, :]).squeeze()

def create_sequences(features, target, window):
    X, y = [], []
    for i in range(len(features) - window):
        X.append(features.iloc[i:i+window].values.flatten())
        y.append(target.iloc[i+window])
    return np.array(X), np.array(y)

# Load and process data
data = load_data(symbol, start_date, end_date)
if data.shape[0] < window_size + 10:
    st.error("Insufficient data. Choose a wider date range.")
    st.stop()

features = data[['Open', 'High', 'Low']]
target = data['Close']
X, y = create_sequences(features, target, window_size)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model training
if st.button("Train Models"):
    linear_model = LinearRegression().fit(X_train_scaled, y_train)
    transformer = TransformerModel(input_dim=64, feature_count=3)
    optimizer = optim.Adam(transformer.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    loader = DataLoader(TensorDataset(torch.tensor(X_train_scaled, dtype=torch.float32),
                                      torch.tensor(y_train, dtype=torch.float32)),
                        batch_size=batch_size, shuffle=True)

    transformer.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(transformer(xb), yb)
            loss.backward()
            optimizer.step()

    # Stack model
    with torch.no_grad():
        tr_train = transformer(torch.tensor(X_train_scaled, dtype=torch.float32)).numpy()
        tr_test = transformer(torch.tensor(X_test_scaled, dtype=torch.float32)).numpy()
    lr_train = linear_model.predict(X_train_scaled)
    lr_test = linear_model.predict(X_test_scaled)

    stack_train = np.column_stack([lr_train, tr_train])
    stack_test = np.column_stack([lr_test, tr_test])
    final_model = LinearRegression().fit(stack_train, y_train)

    torch.save(transformer.state_dict(), f"{model_dir}/{symbol}_transformer.pth")
    joblib.dump([scaler, linear_model, final_model], f"{model_dir}/{symbol}_stacked.pkl")
    st.success("Models trained and saved.")

# Load if exists
model_path = f"{model_dir}/{symbol}_stacked.pkl"
transformer_path = f"{model_dir}/{symbol}_transformer.pth"
if os.path.exists(model_path) and os.path.exists(transformer_path):
    transformer = TransformerModel(input_dim=64, feature_count=3)
    transformer.load_state_dict(torch.load(transformer_path))
    transformer.eval()
    scaler, linear_model, final_model = joblib.load(model_path)

    with torch.no_grad():
        tr_test = transformer(torch.tensor(X_test_scaled, dtype=torch.float32)).numpy()
    lr_test = linear_model.predict(X_test_scaled)
    stack_test = np.column_stack([lr_test, tr_test])
    pred = final_model.predict(stack_test)

    st.subheader("Test Results")
    st.write(f"MAE: {mean_absolute_error(y_test, pred):.4f}")
    st.write(f"RMSE: {np.sqrt(mean_squared_error(y_test, pred)):.4f}")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(y_test, label='Actual')
    ax.plot(pred, label='Predicted')
    ax.legend()
    ax.set_title("Test Set Prediction")
    st.pyplot(fig)

    # Forecast
    recent = features[-window_size:].copy()
    future_preds = []
    for _ in range(forecast_days):
        seq = recent.values.flatten().reshape(1, -1)
        seq_scaled = scaler.transform(seq)
        with torch.no_grad():
            tr_out = transformer(torch.tensor(seq_scaled, dtype=torch.float32)).numpy()
        lr_out = linear_model.predict(seq_scaled)
        final_pred = final_model.predict(np.column_stack([lr_out, tr_out]))[0]
        future_preds.append(final_pred)

        new_row = pd.DataFrame([[final_pred] * 3], columns=['Open', 'High', 'Low'])
        recent = pd.concat([recent.iloc[1:], new_row], ignore_index=True)

    future_dates = [data.index[-1] + datetime.timedelta(days=i+1) for i in range(forecast_days)]
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(future_dates, future_preds, marker='o', label='Forecast')
    ax2.set_title(f"{symbol} - {forecast_days} Day Forecast")
    ax2.legend()
    st.pyplot(fig2)
