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

st.title("Stock Price Prediction: Transformer + Linear Regression + ARIMA Ensemble")

# --- USER INPUTS ---
stock_symbol = st.text_input("Stock Symbol", value="AAPL")
start_date = st.date_input("Start Date", value=pd.to_datetime("2013-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("2023-01-01"))
window_size = st.slider("Sequence Window Size", min_value=5, max_value=30, value=10)
epochs = st.slider("Transformer Training Epochs", min_value=5, max_value=100, value=20)
batch_size = st.slider("Batch Size", min_value=8, max_value=256, value=64)
transformer_weight = st.slider("Transformer Model Weight", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
arima_weight = 1.0 - transformer_weight
st.write(f"ARIMA Model Weight (auto): {arima_weight:.2f}")

@st.cache_data(show_spinner=True)
def load_data(symbol, start, end):
    df = yf.download(symbol, start=start, end=end, progress=False)
    if df.empty:
        st.error("No data fetched for this stock symbol and date range.")
        st.stop()
    return df[['Open', 'High', 'Low']], df['Close']

def create_sequences(features, targets, window):
    X, y = [], []
    for i in range(len(features) - window):
        X.append(features.iloc[i:i + window].values.flatten())
        y.append(targets.iloc[i + window])
    return np.array(X), np.array(y)

# Load data
features, target = load_data(stock_symbol, start_date, end_date)
if len(features) < window_size + 20:
    st.error("Not enough data for given window size and date range.")
    st.stop()

# Prepare sequences
X, y = create_sequences(features, target, window_size)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Transformer model definition
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_features, num_layers=1, num_heads=1, ffn_hid_dim=128):
        super().__init__()
        self.pos_encoder = nn.Linear(num_features, input_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=ffn_hid_dim, batch_first=True),
            num_layers=num_layers)
        self.fc_out = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x shape: (batch_size, features)
        batch_size = x.size(0)
        seq_len = window_size
        feat_per_step = x.size(1) // seq_len
        x = x.view(batch_size, seq_len, feat_per_step)  # reshape to (batch, seq_len, features)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.fc_out(x[:, -1, :])
        return x.view(-1)

# Train Transformer function
def train_transformer(model, X_train, y_train, epochs, batch_size):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)
        st.write(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")
    return losses

# Train Transformer Button
if st.button("Train and Predict"):
    with st.spinner("Training models and making predictions..."):
        # Linear Regression model
        lin_reg = LinearRegression()
        lin_reg.fit(X_train_scaled, y_train)

        # Transformer model
        input_dim = 64
        num_features = X_train.shape[1] // window_size
        transformer = TransformerModel(input_dim=input_dim, num_features=num_features)
        training_losses = train_transformer(transformer, X_train_scaled, y_train, epochs, batch_size)
        transformer.eval()

        # Transformer predictions
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
            transformer_preds = transformer(X_test_tensor).numpy()

        # ARIMA model (trained only on y_train)
        arima_model = ARIMA(y_train, order=(2,1,0))
        arima_fit = arima_model.fit()
        arima_preds = arima_fit.forecast(steps=len(y_test))

        # Ensemble predictions (weighted average)
        ensemble_preds = transformer_weight * transformer_preds + arima_weight * arima_preds

        # Metrics
        mse = mean_squared_error(y_test, ensemble_preds)
        mae = mean_absolute_error(y_test, ensemble_preds)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_test - ensemble_preds) / y_test)) * 100

        st.subheader("Ensemble Model Performance")
        st.write(f"MSE: {mse:.4f}")
        st.write(f"MAE: {mae:.4f}")
        st.write(f"RMSE: {rmse:.4f}")
        st.write(f"MAPE: {mape:.2f} %")

        # Plot training loss curve for Transformer
        fig1, ax1 = plt.subplots()
        ax1.plot(training_losses, label='Transformer Training Loss')
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Transformer Training Loss over Epochs")
        ax1.legend()
        st.pyplot(fig1)

        # Plot ensemble predictions only
        fig2, ax2 = plt.subplots()
        ax2.plot(ensemble_preds, label='Ensemble Predictions', color='red')
        ax2.set_title(f"Ensemble Predictions for {stock_symbol}")
        ax2.set_xlabel("Test Data Index")
        ax2.set_ylabel("Price")
        ax2.legend()
        st.pyplot(fig2)
