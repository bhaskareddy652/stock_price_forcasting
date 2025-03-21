import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pandas.tseries.offsets import BDay
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Streamlit app title and description
st.title("Apple Stock Price Prediction App")
st.markdown("""
This app predicts Apple stock prices for the next 30 days using a pre-trained Random Forest model.
The model uses historical stock price data and economic indicators to make predictions.
""")

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Apples_stock price dataset.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%d-%m-%Y %H:%M')
    return df

df = load_data()
st.write("Dataset Preview:", df.head())

# Load the pre-trained Random Forest model
@st.cache_resource
def load_model():
    rf = joblib.load('rf_model.pkl')
    return rf

rf = load_model()
st.write("Random Forest model loaded successfully!")

# Define feature columns (same as during training)
features = ['nasdaq_index', 'sp500_index', 'inflation_rate', 'unemployment_rate', 'interest_rate', 'market_sentiment']

# Aggregate hourly data to daily data
df['date'] = df['timestamp'].dt.date
df_daily = df.groupby('date').mean(numeric_only=True).reset_index()

# Verify that stock_price exists
if 'stock_price' not in df_daily.columns:
    st.error("Target feature 'stock_price' not found in dataset. Please check your data.")
    st.stop()

# Sidebar for user input
st.sidebar.header("Prediction Settings")
prediction_horizon = st.sidebar.slider("Prediction Horizon (days)", 1, 60, 30)

# Generate future timestamps
last_timestamp = df['timestamp'].max()
future_timestamps = pd.date_range(start=last_timestamp + BDay(1), 
                                 periods=prediction_horizon, 
                                 freq=BDay())
future_df = pd.DataFrame({'timestamp': future_timestamps})

# Simulate future features using historical data
last_30_days = df_daily.tail(30)

# Calculate average daily percentage changes
nasdaq_changes = last_30_days['nasdaq_index'].pct_change().dropna()
sp500_changes = last_30_days['sp500_index'].pct_change().dropna()
avg_nasdaq_change = nasdaq_changes.mean()
avg_sp500_change = sp500_changes.mean()

st.write(f"Average daily % change in nasdaq_index: {avg_nasdaq_change * 100:.2f}%")
st.write(f"Average daily % change in sp500_index: {avg_sp500_change * 100:.2f}%")

# Simulate future features
last_row = df_daily.iloc[-1].copy()
future_X = pd.DataFrame([last_row[features]] * prediction_horizon, columns=features)

# Apply the average daily change
future_X['nasdaq_index'] = last_row['nasdaq_index'] * (1 + avg_nasdaq_change) ** np.arange(1, prediction_horizon + 1)
future_X['sp500_index'] = last_row['sp500_index'] * (1 + avg_sp500_change) ** np.arange(1, prediction_horizon + 1)

# Predict future stock prices
future_stock_price = rf.predict(future_X)
future_df['predicted_stock_price'] = future_stock_price

# Visualization
st.subheader("Historical and Predicted Stock Prices")
fig, ax = plt.subplots(figsize=(12, 6))

# Plot historical stock price (last 50 days)
ax.plot(df_daily['date'].tail(50), df_daily['stock_price'].tail(50), 
        label='Historical Stock Price', color='blue', marker='o')

# Plot predicted stock price
future_df['timestamp'] = pd.to_datetime(future_df['timestamp'])
ax.plot(future_df['timestamp'], future_df['predicted_stock_price'], 
        label='Predicted Stock Price', color='orange', linestyle='--')

# Customize the plot
ax.set_title('Historical and Predicted Apple Stock Prices')
ax.set_xlabel('Date')
ax.set_ylabel('Stock Price')
ax.legend()
ax.grid(True)
ax.tick_params(axis='x', rotation=45)

# Display the plot in Streamlit
st.pyplot(fig)

# Display the predictions as a table
st.subheader("Predicted Stock Prices for the Next {} Days".format(prediction_horizon))
st.write(future_df[['timestamp', 'predicted_stock_price']])

# Download predictions as CSV
csv = future_df.to_csv(index=False)
st.download_button(
    label="Download Predictions as CSV",
    data=csv,
    file_name='apple_stock_predictions.csv',
    mime='text/csv'
)