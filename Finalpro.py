import streamlit as st
from streamlit_option_menu import option_menu
import plotly.express as px
import plotly.graph_objs as go
# import yfinance as yf
from stocknews import StockNews
import pandas as pd
import re
import requests
import os
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np

# ------------------- CONFIG -------------------
st.set_page_config(page_title="InvestIQ", page_icon="📈", layout="wide")

# ------------------- SESSION STATE INIT -------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# ------------------- LOGIN PAGE -------------------
def login_ui():
    st.title("Login to InvestIQ")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "admin" and password == "invest123":
            st.session_state.logged_in = True
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid credentials!")
           
# ------------------- FETCH FUNCTION -------------------
def fetch_stock_dataset(ticker, start_date, end_date):  
    # Folder to store all CSV files
    folder_name = "stock_data"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    for tic in ticker:
        # Fetch data for each ticker
        api_key="a6a840a1a69946c1a3b6fea7388cfd18"
        try:
            url = "https://api.twelvedata.com/time_series"
            params = {
                "symbol": ticker,
                "interval": "1day",
                "start_date": start_date,
                "end_date": end_date,
                "apikey":api_key,
                "outputsize": 5000  # optional, controls how much data
            }

            response = requests.get(url, params=params)
            data = response.json()

            if "values" not in data:
                st.error("⚠️ Failed to fetch data. Check API limit or symbol.")
                return None

            df = pd.DataFrame(data["values"])
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.rename(columns={
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
                "datetime": "Date"
            })

            # df = df.sort_values("Date").reset_index(drop=True)
            df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
            df = df.sort_values("Date").reset_index(drop=True)

            # Save data in a separate file for each ticker in the same folder
            file_name = os.path.join(folder_name, f"{ticker}_stock_data.csv")
            df.to_csv(file_name, index=False)
            print(f"Data for {ticker} saved in file: {file_name}")
            return df
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return None

#
# ------------------- FETCH FUNCTION -------------------
# def fetch_stock_dataset(ticker, start_date, end_date):
#     try:
#         stock = yf.Ticker(ticker)
#         df = stock.history(start=start_date, end=end_date)
#         if df.empty:
#             st.error("⚠️ No data found for this ticker and date range.")
#             return None
#         df.reset_index(inplace=True)
#         return df
#     except Exception as e:
#         st.error(f"Error fetching data: {e}")
#         return None


# ------------------- MAIN APP -------------------
if not st.session_state.logged_in:
    login_ui()
else:
    st.markdown("""
        <div style='text-align: center; background: linear-gradient(300deg, #89f7fe, #66a6ff); 
            padding: 2rem; border-radius: 15px;'>
            <h1>InvestIQ</h1>
            <h3>Your Smart Stock Market Companion</h3>
        </div>
    """, unsafe_allow_html=True)

    with st.sidebar:
        st.markdown("## 📊 Stock Dashboard")
        ticker = st.text_input("📌 Stock Ticker", placeholder= "Enter Stock Name : ",key="ticker_input", help="Example: AAPL, TSLA, MSFT")
        start_date = st.date_input("📅  Select Start Date")
        end_date = st.date_input("📅 Select End Date")
        selected_option = option_menu(
            "Navigation",
            ["📥 Fetch Stock Data", "📊 Visualization", "📰 Stock News","🤖 Stock Price Predict","🚪 Logout"],
            icons=["cloud-download", "bar-chart-line", "newspaper","box-arrow-right"],
            default_index=0
        )
     #for logout
        if selected_option == "🚪 Logout":
            st.session_state.logged_in = False
            st.success("✅ Logged out successfully.")
            st.rerun()


# Fetch Stock Data Page
    if selected_option == "📥 Fetch Stock Data":
        st.subheader(f"📊 Fetching data for {ticker}")
        if st.button("🚀 Fetch Data"):
            dataset = fetch_stock_dataset(ticker, start_date, end_date)
            if dataset is not None:
                st.success(f"Data fetched for {ticker}")
                st.write(dataset)

# Visualization page
    elif selected_option == "📊 Visualization":
        vis_option = option_menu(
            "Choose Visualization",
            ["📈 Line Chart", "🕯️ Candlestick", "📊 Volume","📈 Live Stock Visualization"],
            icons=["graph-up", "graph-down", "bar-chart"]
        )
        dataset = fetch_stock_dataset(ticker, start_date, end_date)
        if dataset is not None:
            if vis_option == "📈 Line Chart":
                fig = px.line(dataset, x="Date", y="Close", title="Closing Price Trend")
                st.plotly_chart(fig)
            elif vis_option == "🕯️ Candlestick":
                fig = go.Figure(data=[go.Candlestick(
                    x=dataset["Date"],
                    open=dataset["Open"],
                    high=dataset["High"],
                    low=dataset["Low"],
                    close=dataset["Close"]
                )])
                st.plotly_chart(fig)
            elif vis_option == "📊 Volume":
                fig = px.bar(dataset, x="Date", y="Volume", title="Trading Volume")
                st.plotly_chart(fig)
            elif vis_option == "📈 Live Stock Visualization":
                # if ticker:
                #     try:
                #         stock = yf.Ticker(ticker)
                #         hist = stock.history(period="1mo", interval="1d")

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=dataset.index, y=dataset['Close'], mode='lines+markers', name='Close'))
                        fig.update_layout(
                            title=f"{ticker.upper()} Stock Price - Last 10 Days",
                            xaxis_title="Date",
                            yaxis_title="Close Price (USD)",
                            template="plotly_white"
                        )

                        st.plotly_chart(fig, use_container_width=True)
                    # except Exception as e:
                    #     st.error(f"⚠️ Failed to load data for {ticker}. Error: {e}")

#------------ News Page-----------------
    elif selected_option == "📰 Stock News":
        news_option = option_menu(
                    "News Option",
                    ["📢 Latest News", "🔍 Sentiment Analysis"],
                    icons=["newspaper", "search"]
                )
        try:
                if news_option == "📢 Latest News":
                    st.subheader(f"📰 News for {ticker}")
                    sn = StockNews(ticker.upper(), save_news=False)
                    news_df = sn.read_rss()
                    if news_df.empty:
                        st.info("No news found.")
                    for i in range(min(5, len(news_df))):
                        st.markdown(f"### {news_df['title'][i]}")
                        st.write(news_df['published'][i])  
                        st.write(news_df['summary'][i])     
                # news sentiment Analysis
                elif news_option == "🔍 Sentiment Analysis":
                        st.subheader("📊 News Sentiment Analysis")
                        sn = StockNews(ticker.upper(), save_news=False)
                        news_df = sn.read_rss()
                        sentiments = []
                        for summary in news_df['summary']:
                            blob = TextBlob(summary)
                            polarity = blob.sentiment.polarity
                            if polarity > 0:
                                sentiments.append("Positive")
                            elif polarity < 0:
                                sentiments.append("Negative")
                            else:
                                sentiments.append("Neutral")
                        news_df['sentiment'] = sentiments
                        # news sentiment visualization
                        sentiment_counts = news_df['sentiment'].value_counts().to_dict()
                        positive = sentiment_counts.get("Positive", 0)
                        neutral = sentiment_counts.get("Neutral", 0)
                        negative = sentiment_counts.get("Negative", 0)

                        polarity_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
                        scores = [polarity_map.get(sent, 0) for sent in news_df['sentiment']]
                        avg_score = sum(scores) / len(scores) if scores else 0
                        sentiment_msg = "😃 Positive" if avg_score > 0 else "😐 Neutral" if avg_score == 0 else "😞 Negative"

                        st.metric("Average Sentiment Score", f"{avg_score:.2f}", delta=sentiment_msg)
                        
                        fig = px.pie(
                            names=["Positive", "Neutral", "Negative"],
                            values=[positive, neutral, negative],
                            title="🧠 Sentiment Distribution",
                            color_discrete_sequence=px.colors.qualitative.Set3
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        #Table of news sentiment
                        try:
                                # Add emoji labels to sentiment
                                styled_df = news_df[['title', 'published', 'sentiment']].copy().head(10)
                                styled_df['sentiment'] = styled_df['sentiment'].map({
                                    'Positive': '🟢 Positive',
                                    'Neutral': '🟡 Neutral',
                                    'Negative': '🔴 Negative'
                                })

                                st.write("### 🗞️ Top 10 Headlines with Sentiment")
                                st.dataframe(styled_df)

                        except Exception as e:
                            st.error(f"Error displaying styled sentiment data: {e}")

        except Exception as e:
            st.error("Error fetching news:,{e}")

#---------------Predication------------------
    elif selected_option == "🤖 Stock Price Predict":
        st.subheader("📈 Predict Next Day Closing Price")

        dataset = fetch_stock_dataset(ticker, start_date, end_date)

        if dataset is None or len(dataset) < 10:
            st.warning("⚠️ Not enough data for prediction. Minimum 10 days required.")
        else:
            # Convert to numeric
            dataset["Close"] = pd.to_numeric(dataset["Close"], errors="coerce")
            dataset = dataset.dropna(subset=["Close"])

            # Prepare data
            look_back = 10
            close_prices = dataset["Close"].values.reshape(-1, 1)

            scaler = MinMaxScaler()
            scaled_close = scaler.fit_transform(close_prices)

            X, y = [], []
            for i in range(len(scaled_close) - look_back):
                X.append(scaled_close[i:i + look_back])
                y.append(scaled_close[i + look_back])
            X, y = np.array(X), np.array(y)

            # Build LSTM regression model
            model = Sequential([
                LSTM(50, input_shape=(look_back, 1)),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X, y, epochs=10, batch_size=16, verbose=0)

            # Predict next day's closing price
            last_sequence = scaled_close[-look_back:].reshape((1, look_back, 1))
            predicted_scaled = model.predict(last_sequence)[0][0]
            predicted_price = scaler.inverse_transform([[predicted_scaled]])[0][0]

            st.success(f"📉 Predicted Next Close Price for {ticker}: **${predicted_price:.2f}**")

            # Visualization
            import plotly.graph_objs as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dataset["Date"], y=dataset["Close"], mode='lines+markers', name='Close Price'))
            fig.add_trace(go.Scatter(x=[dataset["Date"].iloc[-1]], y=[predicted_price],
                                    mode='markers', marker=dict(size=12, color='red'), name='Predicted Next Close'))
            fig.update_layout(title=f"{ticker} - Closing Price and Next Prediction",
                            xaxis_title="Date", yaxis_title="Price")
            st.plotly_chart(fig, use_container_width=True)

            # Show latest rows
            st.markdown("### 🔍 Latest Data")
            st.write(dataset.tail(10))


#---------------check------------------------

            from sklearn.metrics import mean_squared_error, mean_absolute_error
            import numpy as np

            # Predict on training data
            y_pred = model.predict(X)

            # Inverse transform both predictions and actual values
            y_pred_rescaled = scaler.inverse_transform(y_pred)
            y_actual_rescaled = scaler.inverse_transform(y)

            # Calculate metrics
            mse = mean_squared_error(y_actual_rescaled, y_pred_rescaled)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_actual_rescaled, y_pred_rescaled)

            # Show in Streamlit
            print("### 📏 Model Evaluation Metrics (on Training Data)")
            print(f"✅ **MSE (Mean Squared Error):** {mse:.2f}")
            print(f"✅ **RMSE (Root Mean Squared Error):** {rmse:.2f}")
            print(f"✅ **MAE (Mean Absolute Error):** {mae:.2f}")
            
