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

# Title
st.title("Stock Prediction with Transformer + Linear Regression (Stacked Ensemble)")

# --- INPUTS ---
stock_symbol = st.text_input("Stock Symbol", "AAPL")
start_date = st.date_input("Start Date", pd.to_datetime("2013-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2025-01-01"))
window_size = st.number_input("Window Size (sequence length)", min_value=3, max_value=50, value=10, step=1)
epochs = st.slider("Training Epochs", 1, 100, 20)
batch_size = st.number_input("Batch Size", min_value=8, max_value=256, value=64, step=8)

# --- DATA LOADING ---
@st.cache_data
def load_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end)
    return df[['Open', 'High', 'Low', 'Close']]

df = load_data(stock_symbol, start_date, end_date)
if df.isna().sum().sum() > 0 or len(df) < window_size + 20:
    st.error("Not enough or invalid data. Choose a different stock/date range.")
    st.stop()

features = df[['Open', 'High', 'Low']]
target = df['Close']

def create_sequences(features, targets, window=10):
    X, y, dates = [], [], []
    for i in range(len(features) - window):
        X.append(features.iloc[i:i + window].values.flatten())
        y.append(targets.iloc[i + window])
        dates.append(targets.index[i + window])  # index aligned with y
    return np.array(X), np.array(y), np.array(dates)


X, y, dates = create_sequences(features, target, window_size)

X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
    X, y, dates, test_size=0.2, random_state=42)

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
        x = x.view(-1, window_size, x.size(1) // window_size)  # (batch, seq_len, features)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.fc_out(x[:, -1, :])
        return x.squeeze(-1)

# --- TRAIN FUNCTION ---
def train_transformer(model, X_train, y_train, epochs=20, lr=0.001):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                      torch.tensor(y_train, dtype=torch.float32)),
                        batch_size=batch_size, shuffle=True)
    losses = []
    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        losses.append(avg_loss)
    return losses

# --- TRAIN MODELS ---
train_button = st.button("Train Ensemble")
if train_button:
    input_dim = 64
    num_features = X_train.shape[1] // window_size

    # Train Transformer
    transformer = TransformerModel(input_dim=input_dim, num_features=num_features)
    train_loss = train_transformer(transformer, X_train_scaled, y_train, epochs=epochs)

    # Predict with Transformer
    transformer.eval()
    with torch.no_grad():
        transformer_preds_test = transformer(torch.tensor(X_test_scaled, dtype=torch.float32)).numpy()
        transformer_preds_train = transformer(torch.tensor(X_train_scaled, dtype=torch.float32)).numpy()

    # Linear Regression
    linear_model = LinearRegression()
    linear_model.fit(X_train_scaled, y_train)
    lr_preds_test = linear_model.predict(X_test_scaled)
    lr_preds_train = linear_model.predict(X_train_scaled)

    # --- STACKING ---
    stacked_train = np.column_stack((lr_preds_train, transformer_preds_train))
    stacked_test = np.column_stack((lr_preds_test, transformer_preds_test))

    final_model = LinearRegression()
    final_model.fit(stacked_train, y_train)
    y_pred = final_model.predict(stacked_test)

    # --- METRICS ---
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    st.subheader("📊 Performance Metrics")
    st.write(f"**Final Stacked MSE:** {mse:.4f}")
    st.write(f"**MAE:** {mae:.4f}")
    st.write(f"**RMSE:** {rmse:.4f}")
    st.write(f"**MAPE:** {mape:.2f}%")

# --- PLOT ---
st.subheader("📈 Actual vs Predicted Closing Prices")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(dates_test, y_test, label="Actual", color='blue')
ax.plot(dates_test, y_pred, label="Stacked Prediction", color='orange')
ax.set_title(f"{stock_symbol} - Actual vs Predicted (Stacked Model)")
ax.set_xlabel("Date")
ax.set_ylabel("Closing Price")
ax.legend()
plt.xticks(rotation=45)
st.pyplot(fig)

    # --- Loss Plot ---
    st.subheader("🔁 Training Loss (Transformer)")
    fig2, ax2 = plt.subplots()
    ax2.plot(train_loss)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Transformer Training Loss")
    st.pyplot(fig2)
