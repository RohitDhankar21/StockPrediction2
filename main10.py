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

st.title("Stacked Ensemble: Transformer + Linear Regression on OHLC")

# --- USER INPUT ---
stock_symbol = st.text_input("Stock Symbol", "AAPL")
start_date = st.date_input("Start Date", pd.to_datetime("2013-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2023-01-01"))
window_size = st.slider("Window Size", 5, 50, 10)
epochs = st.slider("Training Epochs (Transformer)", 10, 100, 50)
input_dim = st.selectbox("Transformer Input Dim", [32, 64, 128], index=1)

# --- LOAD DATA ---
@st.cache_data
def load_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end, auto_adjust=True)
    return df

df = load_data(stock_symbol, start_date, end_date)
if df.empty or len(df) < window_size + 10:
    st.warning("Not enough data to create sequences.")
    st.stop()

features = df[['Open', 'High', 'Low']]
target = df['Close']

# --- CREATE SEQUENCES ---
def create_sequences(features, targets, window=10):
    X, y = [], []
    for i in range(len(features) - window):
        X.append(features.iloc[i:i + window].values.flatten())
        y.append(targets.iloc[i + window])
    return np.array(X), np.array(y)

X, y = create_sequences(features, target, window=window_size)

# --- SPLIT & SCALE ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- TRANSFORMER MODEL ---
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_features, num_layers=1, num_heads=1, ffn_hid_dim=128):
        super().__init__()
        self.pos_encoder = nn.Linear(num_features, input_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=ffn_hid_dim, batch_first=True),
            num_layers=num_layers)
        self.fc_out = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = x.view(-1, window_size, x.size(1) // window_size)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.fc_out(x[:, -1, :])
        return x.squeeze(-1)

# --- TRAIN TRANSFORMER ---
@st.cache_resource
def train_transformer_model(model, X_train, y_train, epochs=50, lr=0.001):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    for _ in range(epochs):
        for inputs, targets in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1), targets.view(-1))
            loss.backward()
            optimizer.step()
    return model

# --- TRAIN MODELS ---
if st.button("Train & Predict"):
    # Linear Model
    linear_model = LinearRegression()
    linear_model.fit(X_train_scaled, y_train)
    lr_train_preds = linear_model.predict(X_train_scaled)
    lr_test_preds = linear_model.predict(X_test_scaled)

    # Transformer Model
    num_features = X_train.shape[1] // window_size
    transformer = TransformerModel(input_dim=input_dim, num_features=num_features)
    trained_transformer = train_transformer_model(transformer, X_train_scaled, y_train, epochs=epochs)
    transformer.eval()
    with torch.no_grad():
        transformer_train_preds = transformer(torch.tensor(X_train_scaled, dtype=torch.float32)).numpy()
        transformer_test_preds = transformer(torch.tensor(X_test_scaled, dtype=torch.float32)).numpy()

    # Stack predictions
    stacked_train = np.column_stack((lr_train_preds, transformer_train_preds))
    stacked_test = np.column_stack((lr_test_preds, transformer_test_preds))

    # Meta-model
    meta_model = LinearRegression()
    meta_model.fit(stacked_train, y_train)
    final_preds = meta_model.predict(stacked_test)

    # --- METRICS ---
    mse = mean_squared_error(y_test, final_preds)
    mae = mean_absolute_error(y_test, final_preds)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test - final_preds) / y_test)) * 100

    st.subheader("📊 Evaluation Metrics")
    st.write(f"**MSE:** {mse:.4f}")
    st.write(f"**MAE:** {mae:.4f}")
    st.write(f"**RMSE:** {rmse:.4f}")
    st.write(f"**MAPE:** {mape:.2f}%")

    # --- PLOT ---
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(y_test, label="Actual", alpha=0.7)
    ax.plot(final_preds, label="Stacked Ensemble Prediction", alpha=0.7)
    ax.set_title(f"{stock_symbol}: Actual vs Predicted Close Prices")
    ax.set_xlabel("Test Sample Index")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)
