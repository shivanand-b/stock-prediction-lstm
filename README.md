# AI Stock Prediction Dashboard (LSTM)

A Streamlit-based stock analysis and next-day closing price prediction system using an LSTM deep learning model.

Live Demo: https://lstm-stock-prediction-app.streamlit.app/dashboard

## Features
- Login/Signup using SQLite (`users.db`)
- Stock data download using Yahoo Finance (`yfinance`)
- Professional charts (Plotly): Line / Candlestick + Volume, EMA 20/50, Bollinger Bands, RSI
- LSTM next-day close prediction
- Metrics: RMSE, MAE, MAPE (Approx accuracy = 100 - MAPE)
- Forward Testing (Live Accuracy Log):
  - Save prediction to SQLite (`predictions.db`)
  - Refresh later to fetch actual close and compute live errors

## Project Structure
```text
project/
├── login_app.py
├── requirements.txt
├── README.md
└── pages/
    └── dashboard.py
```

## How to Run
```bash
pip install -r requirements.txt
streamlit run login_app.py
```

## Notes
- This is an educational project (not financial advice).
- `actual_close` in forward test will show only after market close and Yahoo updates the data.
