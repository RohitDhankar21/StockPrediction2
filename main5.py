import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# --------- Transformer Model Definition ---------
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_features, num_layers=1, num_heads=1, ffn_hid_dim=128):
        super().__init__()
        self.pos_encoder = nn.Linear(num_features, input_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=ffn_hid_dim, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(input_dim, 1)

    def forward(self, x):
        seq_len = 10  # fixed window size
        x = x.view(-1, seq_len, x.size(1)//seq_len)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.fc_out(x[:, -1, :])
        return x.view(-1, 1)

# --------- Data Preparation ---------
def create_sequences(features, targets, window=10):
    X, y = [], []
    for i in range(len(features) - window):
        X.append(features.iloc[i:i + window].values.flatten())
        y.append(targets.iloc[i + window])
    return np.array(X), np.array(y)

# --------- Train Transformer ---------
def train_transformer_model(model, X_train, y_train, epochs=10, lr=0.001):
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
            loss = criterion(outputs.view(-1), targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        st.write(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss/len(loader):.6f}")
    return model

# --------- Main Streamlit App ---------
def main():
    st.title("📈 Stock Price Prediction with Transformer + Linear Regression + ARIMA Ensemble")

    st.markdown("""
    This app downloads historical stock data, trains three models (Transformer, Linear Regression, and ARIMA),  
    and combines their predictions using a weighted ensemble.  
    You can select any stock from the S&P 500 list below and see evaluation metrics and prediction plots.
    """)

    # Load ticker list (S&P 500) from Wikipedia
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    sp500_table = pd.read_html(url)[0]
    tickers = sp500_table['Symbol'].tolist()
    company_names = sp500_table['Security'].tolist()

    # Dropdown for stock selection
    selected_ticker = st.selectbox("Select a Stock Symbol (S&P 500)", tickers)
    company_name = company_names[tickers.index(selected_ticker)]
    st.write(f"Selected Company: **{company_name} ({selected_ticker})**")

    # Download stock data
    with st.spinner("Downloading data..."):
        data = yf.download(selected_ticker, start='2013-01-01', end='2023-01-01', auto_adjust=True)
    if data.empty:
        st.error("Failed to download data. Try a different stock symbol.")
        return

    features = data[['Open', 'High', 'Low']]
    target = data['Close']

    # Prepare sequences
    X, y = create_sequences(features, target, window=10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    st.write(f"Dataset: {len(X_train)} training samples, {len(X_test)} test samples")

    # Train Linear Regression
    linear_model = LinearRegression()
    linear_model.fit(X_train_scaled, y_train)

    # Train Transformer model
    input_dim = 64
    num_features = X_train.shape[1] // 10
    transformer_model = TransformerModel(input_dim=input_dim, num_features=num_features)

    if st.button("Train Transformer Model"):
        with st.spinner("Training Transformer..."):
            transformer_model = train_transformer_model(transformer_model, X_train_scaled, y_train, epochs=10)
        st.success("Transformer model training completed!")

    if 'transformer_model_trained' not in st.session_state:
        st.session_state['transformer_model_trained'] = False

    # Save model in session state after training
    if st.button("Save Transformer Model"):
        st.session_state.transformer_model = transformer_model
        st.session_state.transformer_model_trained = True
        st.success("Transformer model saved in session!")

    # Use saved model if available
    if st.session_state.get('transformer_model_trained', False):
        transformer_model = st.session_state.transformer_model
        transformer_model.eval()

        # Get predictions from each model
        with torch.no_grad():
            transformer_test_preds = transformer_model(torch.tensor(X_test_scaled, dtype=torch.float32)).numpy().ravel()
        lr_test_preds = linear_model.predict(X_test_scaled)
        arima_model = ARIMA(y_train, order=(2, 1, 0))
        arima_fitted = arima_model.fit()
        arima_preds = arima_fitted.forecast(steps=len(y_test)).ravel()

        # Ensure proper shape and type for all
        y_test = np.array(y_test).astype(float).ravel()
        transformer_test_preds = np.array(transformer_test_preds).astype(float).ravel()
        arima_preds = np.array(arima_preds).astype(float).ravel()
        lr_test_preds = np.array(lr_test_preds).astype(float).ravel()

        # Ensemble prediction
        transformer_weight = 0.2
        arima_weight = 0.6
        linear_weight = 0.2
        assert abs(transformer_weight + arima_weight + linear_weight - 1.0) < 1e-6

        ensemble_preds = (
            transformer_weight * transformer_test_preds +
            arima_weight * arima_preds +
            linear_weight * lr_test_preds
        ).astype(float).ravel()

        # Evaluation metrics
        mse = mean_squared_error(y_test, ensemble_preds)
        mae = mean_absolute_error(y_test, ensemble_preds)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_test - ensemble_preds) / y_test)) * 100
        r2 = r2_score(y_test, ensemble_preds)

        st.subheader("📊 Evaluation Metrics on Test Set")
        st.markdown(f"""
        - **Mean Squared Error (MSE)**: {mse:.4f}  
        - **Mean Absolute Error (MAE)**: {mae:.4f}  
        - **Root Mean Squared Error (RMSE)**: {rmse:.4f}  
        - **Mean Absolute Percentage Error (MAPE)**: {mape:.2f}%  
        - **R-squared (R²)**: {r2:.4f}
        """)

        # Plot ensemble prediction
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(y_test, label="Actual Close Price")
        ax.plot(ensemble_preds, label="Ensemble Prediction")
        ax.set_title(f"{company_name} ({selected_ticker}) - Actual vs Ensemble Predicted Close Price")
        ax.set_xlabel("Test Sample Index")
        ax.set_ylabel("Price")
        ax.legend()
        st.pyplot(fig)

        # Optionally plot individual model predictions
        if st.checkbox("🔍 Show individual model predictions"):
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            ax2.plot(y_test, label="Actual Close Price")
            ax2.plot(transformer_test_preds, label="Transformer Predictions")
            ax2.plot(arima_preds, label="ARIMA Predictions")
            ax2.plot(lr_test_preds, label="Linear Regression Predictions")
            ax2.set_title(f"{company_name} ({selected_ticker}) - Individual Model Predictions")
            ax2.set_xlabel("Test Sample Index")
            ax2.set_ylabel("Price")
            ax2.legend()
            st.pyplot(fig2)

if __name__ == "__main__":
    main()
