import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# Streamlit App Configuration
st.set_page_config(page_title="Stock Price Forecaster", layout="wide")
st.title("📈 Stock Price Forecasting App")

# Data Processing Function
def process_data(df):
    # Convert and filter timestamps
    df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True)
    df = df.set_index('timestamp').between_time('04:00', '20:00').reset_index()
    
    # Exclude weekends
    df = df[df['timestamp'].dt.dayofweek < 5]
    
    # Feature Engineering
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    for lag in [1, 2, 3]:
        df[f'nasdaq_lag_{lag}'] = df['nasdaq_index'].shift(lag)
        df[f'sp500_lag_{lag}'] = df['sp500_index'].shift(lag)
    
    return df.dropna()

# Model Training and Evaluation
def train_and_evaluate(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'MSE': mean_squared_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred)
    }
    
    return model, metrics, scaler, y_pred

# Streamlit UI
uploaded_file = st.file_uploader("Upload your stock data CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    processed_df = process_data(df)
    
    # Feature Selection
    features = ['nasdaq_index', 'sp500_index', 'inflation_rate', 
               'unemployment_rate', 'interest_rate', 'market_sentiment',
               'hour', 'day_of_week', 'nasdaq_lag_1', 'sp500_lag_1']
    
    X = processed_df[features]
    y = processed_df['stock_price']
    
    # Time-based train-test split (no shuffling)
    test_size = int(len(X) * 0.2)
    X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
    y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]
    
    # Train and evaluate
    model, metrics, scaler, y_pred = train_and_evaluate(X_train, X_test, y_train, y_test)
    
    # Display results
    st.subheader("📊 Model Performance")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE", f"{metrics['MAE']:.2f}")
    col2.metric("MSE", f"{metrics['MSE']:.2f}")
    col3.metric("RMSE", f"{metrics['RMSE']:.2f}")
    col4.metric("R² Score", f"{metrics['R2']:.4f}")
    
    # Plot actual vs predicted
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_test.index, y_test, label='Actual Prices')
    ax.plot(y_test.index, y_pred, label='Predicted Prices', alpha=0.7)
    ax.set_title("Actual vs Predicted Stock Prices")
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.legend()
    st.pyplot(fig)
    
    # Feature Importance
    st.subheader("🔍 Feature Importance")
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    st.bar_chart(importance.set_index('Feature'))
    
    # Forecast next period
    st.subheader("🔮 Next Period Forecast")
    last_data = X.iloc[-1:].copy()
    last_data_scaled = scaler.transform(last_data)
    forecast = model.predict(last_data_scaled)[0]
    
    st.metric("Predicted Price", f"{forecast:.2f}", 
             delta=f"{((forecast - y.iloc[-1])/y.iloc[-1]*100):.2f}% from last price")
    
    # Data summary
    with st.expander("Show processed data"):
        st.dataframe(processed_df.tail())
else:
    st.info("ℹ️ Please upload a CSV file with required columns including 'timestamp' and 'stock_price'")
