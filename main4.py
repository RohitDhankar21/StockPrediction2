import streamlit as st
import yfinance as yf
import pandas as pd
import cufflinks as cf
import datetime
from datetime import date
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import requests
from bs4 import BeautifulSoup
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# ------------------------- UI SETUP -------------------------

st.markdown("""
# 📈 Stock Analysis & Prediction App  
Analyze stock performance, and predict future trends!  

**Features:**  
✔ Stock information (logo, summary, industry, market cap)  
✔ Historical stock price charts
✔ Stock price prediction using Prophet and Ensemble Models
✔ Compare two stocks  
✔ Download stock data  
""")
st.write("---")

st.sidebar.subheader('📊 Query Parameters')
start_date = st.sidebar.date_input("🗕 Start Date", datetime.date(2019, 1, 1))
end_date = st.sidebar.date_input("🗕 End Date", datetime.date(2021, 1, 31))

ticker_list = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/s-and-p-500-companies/master/data/constituents_symbols.txt')
stocks = ticker_list.iloc[:, 0].tolist()
selected_stock = st.sidebar.selectbox("📌 Choose Stock Ticker", stocks)

# ------------------------- STOCK INFO -------------------------

tickerData = yf.Ticker(selected_stock)
tickerDf = tickerData.history(period='1d', start=start_date, end=end_date)

st.header(f'**📊 {tickerData.info.get("longName", "Company Name Not Available")}**')
logo_url = tickerData.info.get('logo_url', None)
if not logo_url:
    company_domain = tickerData.info.get('website', '').replace('http://', '').replace('https://', '').strip('/')
    if company_domain:
        logo_url = f"https://logo.clearbit.com/{company_domain}"
if logo_url:
    st.image(logo_url, width=150)
else:
    st.warning("⚠️ No logo available for this company.")
string_summary = tickerData.info.get('longBusinessSummary', 'No company summary available.')
st.info(string_summary)

st.subheader("📊 Stock Overview")
st.write(f"**Sector:** {tickerData.info.get('sector', 'N/A')} | **Industry:** {tickerData.info.get('industry', 'N/A')}")
st.write(f"**Market Cap:** {tickerData.info.get('marketCap', 'N/A'):,}")
st.write(f"**Current Price:** {tickerData.info.get('currentPrice', 'N/A')} | **Previous Close:** {tickerData.info.get('previousClose', 'N/A')}")
st.write(f"**52-Week High:** {tickerData.info.get('fiftyTwoWeekHigh', 'N/A')} | **52-Week Low:** {tickerData.info.get('fiftyTwoWeekLow', 'N/A')}")
st.write(f"**P/E Ratio:** {tickerData.info.get('trailingPE', 'N/A')}")
dy = tickerData.info.get('dividendYield', 'N/A')
st.write(f"**Dividend Yield:** {dy:.2%}" if isinstance(dy, (int, float)) and dy else "**Dividend Yield:** N/A")
st.write(f"**Beta (Volatility Indicator):** {tickerData.info.get('beta', 'N/A')}")
st.write(f"**Current Volume:** {tickerData.info.get('volume', 'N/A'):,} | **Average Volume:** {tickerData.info.get('averageVolume', 'N/A'):,}")
st.write(f"**50-Day MA:** {tickerData.info.get('fiftyDayAverage', 'N/A')} | **200-Day MA:** {tickerData.info.get('twoHundredDayAverage', 'N/A')}")

# ------------------------- NEWS -------------------------
def fetch_stock_news(ticker):
    url = f'https://newsapi.org/v2/everything?q={ticker}&apiKey=047ad87e36534422b4bf4491b9ac6a71'
    response = requests.get(url)
    if response.status_code == 200:
        articles = response.json().get('articles', [])
        return [{'title': a['title'], 'link': a['url'], 'source': a['source']['name']} for a in articles]
    return []

st.subheader("🗞 Latest News")
news = fetch_stock_news(selected_stock)
if news:
    for article in news[:5]:
        st.markdown(f"**[{article['title']}]({article['link']})**")
        st.write(f"🔗 Source: {article['source']}")
        st.write("---")
else:
    st.warning("⚠️ No news available.")

# ------------------------- PROPHET PREDICTION -------------------------
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
n_years = st.sidebar.slider("🗕 Years of Prediction", 1, 4)
period = n_years * 365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

data = load_data(selected_stock)
st.subheader("📊 Raw Stock Data")
st.write(data.tail())

# Plot
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='Open'))
fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Close'))
fig.update_layout(title="📈 Stock Price Movement", xaxis_rangeslider_visible=True)
st.plotly_chart(fig)

# Prophet
st.subheader("🔮 Stock Price Prediction (Prophet)")
df_train = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
df_train['ds'] = df_train['ds'].dt.tz_localize(None)
m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)
st.write(forecast.tail())
st.plotly_chart(plot_plotly(m, forecast))
st.write(m.plot_components(forecast))

# ------------------------- ENSEMBLE LEARNING PREDICTION -------------------------
st.subheader("🧬 Advanced Prediction (Transformer + Linear Regression + ARIMA)")

# Prepare dataset
def prepare_sequence(data):
    prices = data['Close'].values.astype(np.float32)
    window = 20
    X, y = [], []
    for i in range(len(prices) - window):
        X.append(prices[i:i+window])
        y.append(prices[i+window])
    return np.array(X), np.array(y)

X, y = prepare_sequence(data)
train_size = int(len(X) * 0.8)
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# Linear Regression
lr = LinearRegression().fit(X_train, y_train)
lr_pred = lr.predict(X_test)

# ARIMA
arima = ARIMA(y_train, order=(5,1,0)).fit()
arima_pred = arima.forecast(steps=len(y_test))

# Simple Transformer
class TransformerRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=20, nhead=2), num_layers=2)
        self.fc = nn.Linear(20, 1)
    def forward(self, x):
        x = self.encoder(x)
        return self.fc(x[:, -1, :])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerRegressor().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

X_torch = torch.tensor(X_train[:, :, None], dtype=torch.float32).to(device)
y_torch = torch.tensor(y_train[:, None], dtype=torch.float32).to(device)
for epoch in range(30):
    model.train()
    optimizer.zero_grad()
    out = model(X_torch)
    loss = loss_fn(out, y_torch)
    loss.backward()
    optimizer.step()

model.eval()
X_test_tensor = torch.tensor(X_test[:, :, None], dtype=torch.float32).to(device)
transformer_pred = model(X_test_tensor).cpu().detach().numpy().flatten()

# Combine Predictions
final_pred = (0.4 * lr_pred + 0.3 * arima_pred + 0.3 * transformer_pred)

# Evaluation
mse = mean_squared_error(y_test, final_pred)
mae = mean_absolute_error(y_test, final_pred)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_test - final_pred) / y_test)) * 100

st.write(f"**MSE:** {mse:.4f} | **MAE:** {mae:.4f} | **RMSE:** {rmse:.4f} | **MAPE:** {mape:.2f}%")

# Plot
fig3 = go.Figure()
fig3.add_trace(go.Scatter(y=y_test, mode='lines', name='Actual'))
fig3.add_trace(go.Scatter(y=final_pred, mode='lines', name='Ensemble Prediction'))
fig3.update_layout(title="🧬 Ensemble Model Prediction vs Actual", xaxis_title="Time Step", yaxis_title="Price")
st.plotly_chart(fig3)
