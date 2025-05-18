import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import yfinance as yf

# -- Transformer model definition --
class SimpleTransformer(nn.Module):
    def __init__(self, feature_size=1, num_layers=1, num_heads=1, d_model=64):
        super(SimpleTransformer, self).__init__()
        self.input_proj = nn.Linear(feature_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, src):
        src = self.input_proj(src)
        output = self.transformer(src)
        output = self.decoder(output)
        return output[:, -1, :].squeeze(-1)

# -- Training function --
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
        total_loss = 0
        for xb, yb in loader:
            xb = xb.unsqueeze(-1).to(device)
            yb = yb.to(device).squeeze()
            optimizer.zero_grad()
            outputs = model(xb)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        st.write(f"Epoch {epoch+1}/{epochs} - Training Loss (MSE): {total_loss/len(loader):.6f}")
    return model

# -- Prediction function --
def predict_transformer(model, X):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(device)
    with torch.no_grad():
        outputs = model(X_t)
    return outputs.cpu().numpy()

# -- Data preparation function --
def prepare_stock_data(stock_symbol, seq_len=20, start_date='2013-01-01', end_date='2025-01-01'):
    data = yf.download(stock_symbol, start=start_date, end=end_date)
    prices = data['Close'].values
    if len(prices) < seq_len + 10:
        st.warning(f"Not enough data for {stock_symbol} to create sequences.")
        return None, None
    X, y = [], []
    for i in range(len(prices) - seq_len):
        X.append(prices[i:i+seq_len])
        y.append(prices[i+seq_len])
    X = np.array(X)
    y = np.array(y)
    split_idx = int(0.8 * len(X))
    return (X[:split_idx], y[:split_idx]), (X[split_idx:], y[split_idx:])

# -- Evaluation metrics function --
def evaluate(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mse, mae, rmse, mape

# -- Load stock symbols from URL --
@st.cache_data
def load_stock_symbols():
    url = 'https://raw.githubusercontent.com/dataprofessor/s-and-p-500-companies/master/data/constituents_symbols.txt'
    symbols = pd.read_csv(url, header=None)[0].tolist()
    return symbols

# -- Streamlit app main --
def main():
    st.title("📈 Transformer Stock Price Prediction")
    st.markdown("""
        This app trains a **Transformer model** to predict stock closing prices based on historical data.
        Select a stock symbol, then train the model and see predictions on the test set.
    """)

    symbols = load_stock_symbols()
    stock_symbol = st.selectbox("Select a stock symbol", symbols, index=symbols.index('AAPL') if 'AAPL' in symbols else 0)
    st.write(f"Selected Stock: **{stock_symbol}**")

    seq_len = st.slider("Sequence length (days used for prediction)", min_value=5, max_value=60, value=20)
    epochs = st.slider("Training epochs", min_value=5, max_value=50, value=10)

    data = prepare_stock_data(stock_symbol, seq_len=seq_len)
    if data is None or data[0] is None:
        st.error("Failed to load or prepare data for the selected stock.")
        return

    (X_train, y_train), (X_test, y_test) = data
    st.write(f"Dataset loaded: {len(X_train)} training samples, {len(X_test)} test samples")

    if 'model' not in st.session_state:
        st.session_state.model = None

    if st.button("Train Transformer Model"):
        with st.spinner("Training... this may take a while"):
            model = train_transformer(X_train, y_train, epochs=epochs)
            st.session_state.model = model
        st.success("Training complete!")

    if st.session_state.model is not None:
        model = st.session_state.model

        if st.button("Predict on Test Set"):
            preds = predict_transformer(model, X_test)
            mse, mae, rmse, mape = evaluate(y_test, preds)

            st.subheader("Evaluation Metrics on Test Set")
            st.write(f"- Mean Squared Error (MSE): {mse:.4f} (Lower is better)")
            st.write(f"- Mean Absolute Error (MAE): {mae:.4f} (Average absolute prediction error)")
            st.write(f"- Root Mean Squared Error (RMSE): {rmse:.4f} (Square root of MSE)")
            st.write(f"- Mean Absolute Percentage Error (MAPE): {mape:.2f}% (Average percent error)")

            # Plot true vs predicted
            st.subheader("Prediction Plot")
            plt.figure(figsize=(12, 5))
            plt.plot(y_test, label='True Prices')
            plt.plot(preds, label='Predicted Prices')
            plt.title(f"{stock_symbol} Closing Price Prediction")
            plt.xlabel("Time Step")
            plt.ylabel("Price")
            plt.legend()
            st.pyplot(plt)
            plt.clf()

if __name__ == "__main__":
    main()
