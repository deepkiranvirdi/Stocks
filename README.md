# 📈 InvestIQ – Smart Stock Market Companion

InvestIQ is an interactive **Stock Market Analysis and Prediction web application** built using **Python and Streamlit**. It allows users to fetch historical stock data, visualize price trends, analyze financial news sentiment, and predict the next day's closing price using an **LSTM deep learning model**.

---

## 🚀 Features

* 🔐 **User Login** – Secure login interface for accessing the dashboard.
* 📥 **Stock Data Fetching** – Fetch historical stock data for a selected ticker and date range.
* 📈 **Interactive Visualizations**

  * Closing Price Trend
  * Candlestick Chart
  * Trading Volume
  * Stock Price Visualization
* 📰 **Stock News** – View the latest news related to a selected stock.
* 🧠 **Sentiment Analysis** – Classify financial news as Positive, Neutral, or Negative using TextBlob.
* 🤖 **Stock Price Prediction** – Predict the next-day closing price using an LSTM neural network.
* 📊 **Model Evaluation** – Calculate MSE, RMSE, and MAE metrics for the prediction model.
* 💾 **CSV Storage** – Downloaded stock data is stored locally as CSV files.

---

## 🛠️ Technologies Used

| Technology         | Purpose                               |
| ------------------ | ------------------------------------- |
| Python             | Core programming language             |
| Streamlit          | Web application interface             |
| Pandas             | Data manipulation and analysis        |
| Plotly             | Interactive charts and visualizations |
| Twelve Data API    | Stock market data                     |
| StockNews          | Financial news collection             |
| TextBlob           | News sentiment analysis               |
| Scikit-learn       | Data scaling and evaluation metrics   |
| TensorFlow / Keras | LSTM prediction model                 |
| NumPy              | Numerical computations                |

---

## 📊 Application Workflow

```text
User Login
    ↓
Enter Stock Ticker & Date Range
    ↓
Fetch Historical Stock Data
    ↓
 ┌───────────────┬─────────────────┬──────────────────┐
 ↓               ↓                 ↓
Visualization   Stock News       Price Prediction
 ↓               ↓                 ↓
Charts          Sentiment        LSTM Model
                Analysis             ↓
                                  Next-Day Price
```

---

## 📈 Stock Visualization

InvestIQ provides multiple ways to analyze historical stock performance:

### Line Chart

Displays the closing price trend over the selected period.

### Candlestick Chart

Shows Open, High, Low, and Close prices to understand daily price movements.

### Trading Volume

Visualizes the trading volume across different dates.

---

## 📰 News & Sentiment Analysis

The application retrieves recent stock-related news and analyzes the sentiment of news summaries using **TextBlob**.

Each article is classified as:

* 🟢 Positive
* 🟡 Neutral
* 🔴 Negative

The application also calculates an overall sentiment score and displays the sentiment distribution using an interactive pie chart.

---

## 🤖 Stock Price Prediction

InvestIQ uses an **LSTM (Long Short-Term Memory)** neural network to predict the next day's closing price.

### Prediction Process

1. Collect historical closing prices.
2. Scale the prices using `MinMaxScaler`.
3. Use the previous **10 days** as the input sequence.
4. Train an LSTM model.
5. Predict the next closing price.
6. Convert the prediction back to the original price scale.

The model uses:

* LSTM layer with 50 units
* Dense output layer
* Adam optimizer
* Mean Squared Error loss
* 10 training epochs
* Batch size of 16

---

## 📏 Model Evaluation

The project evaluates the model using:

* **MSE** – Mean Squared Error
* **RMSE** – Root Mean Squared Error
* **MAE** – Mean Absolute Error

These metrics help measure the difference between the predicted and actual stock prices.

---

`

---

## ⚙️ Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/InvestIQ.git
cd InvestIQ
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the application

```bash
streamlit run Finalpro.py
```

The application will open in your browser.

---

## 📦 Required Libraries

Create a `requirements.txt` file containing:

```text
streamlit
streamlit-option-menu
plotly
stocknews
pandas
requests
textblob
scikit-learn
tensorflow
numpy
```

---

## 🔑 Login

The current application includes a login interface.

**Username:**

```text
admin
```

**Password:**

```text
invest123
```

> ⚠️ For a production application, credentials should not be hard-coded. Environment variables or a proper authentication system should be used instead.

---

## 🔌 API Integration

The application uses the **Twelve Data API** to retrieve historical stock market data.

The API provides:

* Open price
* High price
* Low price
* Closing price
* Trading volume
* Date/time information

> ⚠️ Before publishing the project publicly, move the API key out of the source code and store it securely using environment variables or Streamlit secrets.

---

## 🎯 Project Objective

The objective of InvestIQ is to demonstrate how **data analysis, financial news processing, interactive visualization, and machine learning** can be combined into a single stock market analytics application.

It provides users with a simple interface to explore historical market data, understand news sentiment, and experiment with stock price prediction.

---

## 🔮 Future Enhancements

* Real-time stock price updates
* Portfolio tracking
* Multiple-stock comparison
* Buy/Sell signal generation
* Advanced technical indicators such as RSI and MACD
* Improved model evaluation using separate training and testing datasets
* More advanced forecasting models
* Secure user authentication
* Cloud deployment

---

## ⚠️ Disclaimer

This project is developed for **educational and analytical purposes only**. Stock price predictions are experimental and should not be considered financial advice or used as a guaranteed basis for investment decisions.

---

## 👩‍💻 Author

**Deepkiran Kaur Virdi**

Built as a Python-based data analytics and machine learning project.
