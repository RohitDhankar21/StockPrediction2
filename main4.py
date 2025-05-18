import streamlit as st
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# -- Transformer model definition --
class SimpleTransformer(nn.Module):
    def __init__(self, feature_size=1, num_layers=1, num_heads=1, d_model=64):
        super(SimpleTransformer, self).__init__()
        self.input_proj = nn.Linear(feature_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, src):
        # src shape: (batch, seq_len, feature_size)
        src = self.input_proj(src)
        output = self.transformer(src)  # (batch, seq_len, d_model)
        output = self.decoder(output)   # (batch, seq_len, 1)
        return output[:, -1, :].squeeze(-1)  # predict last timestep, shape (batch,)

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
            xb = xb.unsqueeze(-1).to(device)  # (batch, seq_len, 1)
            yb = yb.to(device).squeeze()      # (batch,)
            optimizer.zero_grad()
            outputs = model(xb)                # (batch,)
            loss = criterion(outputs, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        st.write(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.6f}")
    return model

# -- Prediction function --
def predict_transformer(model, X):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(device)  # (batch, seq_len, 1)
    with torch.no_grad():
        outputs = model(X_t)  # (batch,)
    return outputs.cpu().numpy()

# -- Prepare dummy data (replace with your own stock data loading and preprocessing) --
def generate_dummy_data(seq_len=20, n_samples=300):
    # Generate a noisy sine wave dataset
    x = np.linspace(0, 100, n_samples + seq_len)
    data = np.sin(x) + 0.1 * np.random.randn(len(x))
    X, y = [], []
    for i in range(n_samples):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

# -- Streamlit app --
def main():
    st.title("Transformer Stock Price Prediction Demo")

    seq_len = 20
    X, y = generate_dummy_data(seq_len=seq_len, n_samples=300)

    # Split train/test
    split_idx = int(0.8 * len(X))
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]

    st.write(f"Dataset: {len(X_train)} training samples, {len(X_test)} test samples")

    model = None
    if 'transformer_model' not in st.session_state:
        st.session_state.transformer_model = None

    if st.button("Train Transformer Model"):
        with st.spinner("Training transformer model..."):
            model = train_transformer(X_train, y_train, epochs=10)
            st.session_state.transformer_model = model
        st.success("Transformer model trained!")

    if st.session_state.transformer_model is not None:
        model = st.session_state.transformer_model

        if st.button("Predict on Test Set"):
            preds = predict_transformer(model, X_test)

            mse = mean_squared_error(y_test, preds)
            st.write(f"Test MSE: {mse:.6f}")

            # Plot true vs predicted
            plt.figure(figsize=(10,4))
            plt.plot(y_test, label='True')
            plt.plot(preds, label='Predicted')
            plt.legend()
            plt.title("Transformer Model Predictions on Test Set")
            st.pyplot(plt)
            plt.clf()

if __name__ == "__main__":
    main()
