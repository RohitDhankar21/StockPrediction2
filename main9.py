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
import pmdarima as pm
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

st.title("Stock Price Prediction with Multiple Models")

# --- PARAMETERS ---
stock_symbol = st.text_input("Stock Symbol", "AAPL")
start_date = st.date_input("Start Date", pd.to_datetime("2013-01-01"))
end_date = st.date_input("End Date", pd.to_datetime("2025-01-01"))
window_size = st.number_input("Window Size (sequence length)", min_value=3, max_value=50, value=10, step=1)
epochs = st.slider("Training Epochs", 1, 100, 20)
batch_size = st.number_input("Batch Size", min_value=8, max_value=256, value=64, step=8)

# --- LOAD DATA ---
@st.cache_data
def load_data(symbol, start, end):
    data = yf.download(symbol, start=start, end=end)
    return data['Close']

prices_series = load_data(stock_symbol, start_date, end_date)
prices = prices_series.values

if len(prices) < window_size + 10:
    st.error("Not enough data to create sequences. Please choose different dates or stock.")
    st.stop()

# --- CREATE SEQUENCES ---
def create_sequences(data, window):
    X, y = [], []
    for i in range(len(data) - window):
        X.append(data[i:i + window])
        y.append(data[i + window])
    return np.array(X), np.array(y)

X, y = create_sequences(prices, window_size)

# --- TRAIN-TEST SPLIT ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Store indices for ARIMA model
train_indices, test_indices = train_test_split(range(len(X)), test_size=0.2, random_state=42)
test_indices = sorted(test_indices)  # Sort for sequential prediction

# --- SCALE DATA ---
scaler_X = StandardScaler()
scaler_y = StandardScaler()

# Scale X (flatten first then reshape)
X_train_scaled = scaler_X.fit_transform(X_train.reshape(-1, window_size))
X_test_scaled = scaler_X.transform(X_test.reshape(-1, window_size))

# Scale y separately
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

# Reshape back to 3D for transformer input
X_train_scaled = X_train_scaled.reshape(-1, window_size, 1)
X_test_scaled = X_test_scaled.reshape(-1, window_size, 1)

# --- TRANSFORMER MODEL ---
class TransformerModel(nn.Module):
    def __init__(self, input_dim=1, num_layers=1, num_heads=1, ffn_hid_dim=128):
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
        x = x.permute(1, 0, 2)  # [seq_len, batch, features]
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # [batch, seq_len, features]
        return self.fc_out(x[:, -1, :]).squeeze(-1)

# Initialize model, criterion, optimizer
model = TransformerModel(input_dim=1)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Prepare data tensors and dataloader
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# --- TRAINING ---
loss_progress = []

def train_model(model, loader, optimizer, criterion, epochs):
    model.train()
    epoch_display = st.empty()  # placeholder to update the line

    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(loader)
        loss_progress.append(avg_loss)
        epoch_display.text(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")  # update single line

# --- ARIMA MODEL FUNCTION ---
def fit_arima_model(series, test_indices):
    # Use pmdarima to automatically find best parameters
    train_data = np.delete(series, test_indices)
    
    with st.spinner("Finding optimal ARIMA parameters..."):
        auto_arima = pm.auto_arima(
            train_data,
            start_p=1, start_q=1,
            max_p=3, max_q=3, m=1,
            d=None, seasonal=False,
            trace=False,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True
        )
    
    st.write(f"Selected ARIMA model: ARIMA{auto_arima.order}")
    
    # Forecast for test set
    forecasts = []
    
    # Sort test indices to ensure sequential prediction
    test_indices_sorted = sorted(test_indices)
    history = list(train_data)
    
    with st.spinner("Generating ARIMA forecasts..."):
        progress_bar = st.progress(0)
        for i, idx in enumerate(test_indices_sorted):
            # Fit model on available history
            model = ARIMA(history, order=auto_arima.order)
            model_fit = model.fit()
            
            # Make prediction
            yhat = model_fit.forecast(steps=1)[0]
            forecasts.append(yhat)
            
            # Update history with actual value for next prediction
            history.append(series[idx])
            
            # Update progress
            progress_bar.progress((i + 1) / len(test_indices_sorted))
    
    # Reorder forecasts to match test indices order
    ordered_forecasts = np.zeros(len(test_indices))
    for i, idx in enumerate(test_indices):
        position = test_indices_sorted.index(idx)
        ordered_forecasts[i] = forecasts[position]
    
    return ordered_forecasts

# --- TRAIN BUTTON ---
train_button = st.button("Train Models")
if train_button:
    # Train transformer model
    with st.expander("Transformer Model Training", expanded=True):
        st.subheader("Training Transformer Model")
        train_model(model, train_loader, optimizer, criterion, epochs)

    # --- TRANSFORMER PREDICTIONS ---
    model.eval()
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    with torch.no_grad():
        transformer_preds_scaled = model(X_test_tensor).numpy()

    # Inverse transform to original scale
    transformer_preds = scaler_y.inverse_transform(transformer_preds_scaled.reshape(-1, 1)).flatten()

    # --- LINEAR REGRESSION MODEL ---
    with st.expander("Linear Regression Model", expanded=True):
        st.subheader("Training Linear Regression Model")
        linear_model = LinearRegression()
        linear_model.fit(X_train_scaled.reshape(-1, window_size), y_train_scaled)
        linear_preds_scaled = linear_model.predict(X_test_scaled.reshape(-1, window_size))
        linear_preds = scaler_y.inverse_transform(linear_preds_scaled.reshape(-1, 1)).flatten()

    # --- ARIMA MODEL ---
    with st.expander("ARIMA Model", expanded=True):
        st.subheader("Training ARIMA Model")
        arima_preds = fit_arima_model(prices, test_indices)

    # --- ENSEMBLE (Transformer + Linear Regression) ---
    ensemble1_preds = (transformer_preds + linear_preds) / 2

    # --- ENSEMBLE (Transformer + ARIMA) ---
    ensemble2_preds = (transformer_preds + arima_preds) / 2

    # --- METRICS CALCULATION ---
    def calculate_metrics(actual, predicted):
        mse = mean_squared_error(actual, predicted)
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100
        return {
            "MSE": mse,
            "MAE": mae,
            "RMSE": rmse,
            "MAPE": mape
        }

    # Calculate metrics for both ensembles
    metrics1 = calculate_metrics(y_test, ensemble1_preds)
    metrics2 = calculate_metrics(y_test, ensemble2_preds)

    # --- DISPLAY RESULTS ---
    st.header("Model Performance")
    
    # Two columns for the two ensembles
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Transformer + Linear Regression Ensemble")
        st.write(f"**MSE:** {metrics1['MSE']:.4f}")
        st.write(f"**MAE:** {metrics1['MAE']:.4f}")
        st.write(f"**RMSE:** {metrics1['RMSE']:.4f}")
        st.write(f"**MAPE:** {metrics1['MAPE']:.2f}%")
        
        # Plot first ensemble
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(y_test, label='Actual Prices', color='blue')
        ax1.plot(ensemble1_preds, label='Ensemble Predictions', color='red')
        ax1.set_title(f"Transformer + Linear Regression Ensemble for {stock_symbol}")
        ax1.set_xlabel("Test Sample Index")
        ax1.set_ylabel("Price")
        ax1.legend()
        st.pyplot(fig1)
    
    with col2:
        st.subheader("Transformer + ARIMA Ensemble")
        st.write(f"**MSE:** {metrics2['MSE']:.4f}")
        st.write(f"**MAE:** {metrics2['MAE']:.4f}")
        st.write(f"**RMSE:** {metrics2['RMSE']:.4f}")
        st.write(f"**MAPE:** {metrics2['MAPE']:.2f}%")
        
        # Plot second ensemble
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        ax2.plot(y_test, label='Actual Prices', color='blue')
        ax2.plot(ensemble2_preds, label='Ensemble Predictions', color='green')
        ax2.set_title(f"Transformer + ARIMA Ensemble for {stock_symbol}")
        ax2.set_xlabel("Test Sample Index")
        ax2.set_ylabel("Price")
        ax2.legend()
        st.pyplot(fig2)
    
    # Plot training loss curve for transformer
    fig3, ax3 = plt.subplots()
    ax3.plot(loss_progress, label="Training Loss")
    ax3.set_title("Transformer Training Loss Over Epochs")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Loss")
    st.pyplot(fig3)
    
    # Compare models
    st.header("Model Comparison")
    comparison_df = pd.DataFrame({
        "Transformer + Linear Regression": [metrics1["MSE"], metrics1["MAE"], metrics1["RMSE"], metrics1["MAPE"]],
        "Transformer + ARIMA": [metrics2["MSE"], metrics2["MAE"], metrics2["RMSE"], metrics2["MAPE"]]
    }, index=["MSE", "MAE", "RMSE", "MAPE"])
    
    st.dataframe(comparison_df)
    
    # Visualization of metric comparison
    fig4, ax4 = plt.subplots(figsize=(10, 6))
    comparison_df.iloc[:3].plot(kind='bar', ax=ax4)  # Exclude MAPE as it's on a different scale
    ax4.set_title("Error Metrics Comparison")
    ax4.set_ylabel("Error Value")
    ax4.legend(title="Model")
    st.pyplot(fig4)
    
    # MAPE comparison (separate because of scale difference)
    fig5, ax5 = plt.subplots(figsize=(8, 4))
    comparison_df.loc[["MAPE"]].plot(kind='bar', ax=ax5, color=['#1f77b4', '#2ca02c'])
    ax5.set_title("MAPE Comparison")
    ax5.set_ylabel("MAPE (%)")
    ax5.legend(title="Model")
    st.pyplot(fig5)
