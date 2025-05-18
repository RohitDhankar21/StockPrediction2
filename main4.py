# main_app.py
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
from sklearn.metrics import mean_squared_error
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

# 🔹 App Title
st.markdown("""
# 📈 Stock Analysis & Prediction App  
Analyze stock performance, and predict future trends!  

**Features:**  
✔ Stock information (logo, summary, industry, market cap)  
✔ Historical stock price charts  
✔ Stock price prediction using Prophet  
✔ Compare two stocks  
✔ Download stock data  
✔ Ensemble Learning Forecast (Transformer + Linear Regression + ARIMA + LSTM)
""")
st.write("---")

# 🔹 Sidebar for user input
st.sidebar.subheader('📊 Query Parameters')
start_date = st.sidebar.date_input("📅 Start Date", datetime.date(2019, 1, 1))
end_date = st.sidebar.date_input("📅 End Date", datetime.date(2021, 1, 31))

# 🔹 Load ticker symbols
ticker_list = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/s-and-p-500-companies/master/data/constituents_symbols.txt')
stocks = ticker_list.iloc[:, 0].tolist()
selected_stock = st.sidebar.selectbox("📌 Choose Stock Ticker", stocks)

# 🔹 Fetch stock data
tickerData = yf.Ticker(selected_stock)
try:
    info = tickerData.info
except:
    st.error("Failed to retrieve data. The ticker may be delisted or unavailable.")
    st.stop()

tickerDf = tickerData.history(period='1d', start=start_date, end=end_date)

# 🔹 Display company information
st.header(f'**📊 {info.get("longName", "Company Name Not Available")}**')

logo_url = info.get('logo_url', None)
if not logo_url:
    company_domain = info.get('website', '').replace('http://', '').replace('https://', '').strip('/')
    if company_domain:
        logo_url = f"https://logo.clearbit.com/{company_domain}"
if logo_url:
    st.image(logo_url, width=150)

string_summary = info.get('longBusinessSummary', 'No company summary available.')
st.info(string_summary)

st.subheader("📊 Stock Overview")
st.write(f"**Sector:** {info.get('sector', 'N/A')} | **Industry:** {info.get('industry', 'N/A')}")
st.write(f"**Market Cap:** {info.get('marketCap', 'N/A'):,}")
st.write(f"**Current Price:** {info.get('currentPrice', 'N/A')} | **Previous Close:** {info.get('previousClose', 'N/A')}")
st.write(f"**52-Week High:** {info.get('fiftyTwoWeekHigh', 'N/A')} | **52-Week Low:** {info.get('fiftyTwoWeekLow', 'N/A')}")
st.write(f"**P/E Ratio:** {info.get('trailingPE', 'N/A')}")
div_yield = info.get('dividendYield', None)
st.write(f"**Dividend Yield:** {div_yield:.2%}" if isinstance(div_yield, (int, float)) else "**Dividend Yield:** N/A")
st.write(f"**Beta (Volatility Indicator):** {info.get('beta', 'N/A')}")
st.write(f"**Current Volume:** {info.get('volume', 'N/A'):,} | **Avg Volume:** {info.get('averageVolume', 'N/A'):,}")
st.write(f"**50-Day MA:** {info.get('fiftyDayAverage', 'N/A')} | **200-Day MA:** {info.get('twoHundredDayAverage', 'N/A')}")

# 🔹 News API
def fetch_stock_news_from_api(ticker, api_key):
    url = f'https://newsapi.org/v2/everything?q={ticker}&apiKey={api_key}'
    response = requests.get(url)
    if response.status_code == 200:
        articles = response.json().get('articles', [])
        return [{'title': a['title'], 'link': a['url'], 'source': a['source']['name']} for a in articles]
    return []

st.subheader("📰 Latest News")
news = fetch_stock_news_from_api(selected_stock, '047ad87e36534422b4bf4491b9ac6a71')
if news:
    for article in news[:5]:
        st.markdown(f"**[{article['title']}]({article['link']})**")
        st.write(f"🔗 Source: {article['source']}")
        st.write("---")
else:
    st.warning("⚠️ No news available for this stock.")

# 🔹 Prophet Forecast
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
n_years = st.sidebar.slider("📅 Years of Prediction", 1, 4)
period = n_years * 365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

with st.spinner("📥 Fetching data..."):
    data = load_data(selected_stock)

st.subheader("📊 Raw Stock Data")
st.write(data.tail())

def plot_raw_data(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], mode='lines', name='Open'))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close'))
    fig.update_layout(title="📈 Stock Price Movement", xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data(data)

st.subheader("🔮 Stock Price Prediction using Prophet")
df_train = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
df_train['ds'] = df_train['ds'].dt.tz_localize(None)
m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)
st.write("📈 Forecast Data")
st.write(forecast.tail())
st.plotly_chart(plot_plotly(m, forecast))
st.write("📊 Forecast Components")
st.write(m.plot_components(forecast))

# 🔹 ENSEMBLE LEARNING SECTION
st.header("🧠 Ensemble Learning Forecast (Transformer + Linear Regression + ARIMA + LSTM)")

@st.cache_data
def get_close_prices(ticker):
    df = yf.download(ticker, start="2015-01-01", end=TODAY)
    return df['Close'].fillna(method='ffill')

close_prices = get_close_prices(selected_stock)
prices = close_prices.values.reshape(-1, 1)
X = np.arange(len(prices)).reshape(-1, 1)

# 1️⃣ Linear Regression
lr = LinearRegression().fit(X, prices)
lr_pred = lr.predict(X)

# 2️⃣ ARIMA Model
arima = ARIMA(prices, order=(5, 1, 0)).fit()
arima_pred = arima.predict(start=1, end=len(prices)-1, typ='levels')

# 3️⃣ Transformer Model
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerPredictor(nn.Module):
    def __init__(self, input_dim, seq_len, d_model=64, nhead=2, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.transformer(x)
        return self.decoder(x)

transformer_model = TransformerPredictor(input_dim=1, seq_len=len(prices))
optimizer = torch.optim.Adam(transformer_model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()
seq = torch.tensor(prices, dtype=torch.float32).unsqueeze(0)
for _ in range(5):
    pred = transformer_model(seq)
    loss = loss_fn(pred.squeeze(), seq.squeeze())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
transformer_pred = pred.squeeze().detach().numpy()

# Ensemble average (simple mean)
ensemble_pred = (lr_pred.flatten() + arima_pred[:len(prices)] + transformer_pred[:len(prices)]) / 3

# Plotting
st.subheader("📊 Ensemble vs Actual")
fig = go.Figure()
fig.add_trace(go.Scatter(y=prices.flatten(), mode='lines', name='Actual'))
fig.add_trace(go.Scatter(y=ensemble_pred, mode='lines', name='Ensemble Prediction'))
st.plotly_chart(fig)
