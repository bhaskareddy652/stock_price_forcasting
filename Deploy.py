import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime

st.title("📈 Stock Price Forecasting App")

# Configuration
st.sidebar.header("Settings")
test_size = st.sidebar.slider("Test Size Ratio", 0.1, 0.5, 0.2, 0.05)
forecast_days = st.sidebar.number_input("Forecast Days", 1, 90, 30)

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        # Check if required columns exist
        if 'stock_price' not in df.columns:
            st.error("❌ Error: 'stock_price' column not found in dataset")
            st.stop()
        
        # Auto-detect and process date columns
        date_cols = [col for col in df.columns if any(x in col.lower() for x in ['date', 'time'])]
        for col in date_cols:
            try:
                df[col] = pd.to_datetime(df[col])
                df[f'{col}_year'] = df[col].dt.year
                df[f'{col}_month'] = df[col].dt.month
                df[f'{col}_day'] = df[col].dt.day
                df[f'{col}_dayofweek'] = df[col].dt.dayofweek
                df = df.drop(col, axis=1)
            except:
                st.warning(f"⚠️ Could not parse column '{col}' as datetime")

        st.subheader("📊 Dataset Preview")
        st.write(df.head())

        # Feature selection (exclude target)
        available_features = [col for col in df.columns if col != 'stock_price']
        if not available_features:
            st.error("❌ No features available for prediction")
            st.stop()

        features = st.multiselect(
            "Select features for prediction",
            available_features,
            default=available_features[:min(5, len(available_features))]
        )

        # Prepare data
        X = df[features]
        y = df['stock_price']

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=42,
            shuffle=False  # Important for time series data
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Model training
        with st.spinner('Training model...'):
            try:
                model = RandomForestRegressor(
                    n_estimators=150,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
                model.fit(X_train_scaled, y_train)
            except Exception as e:
                st.error(f"❌ Model training failed: {str(e)}")
                st.stop()

        # Evaluation
        y_pred = model.predict(X_test_scaled)
        
        st.subheader("📈 Model Performance")
        col1, col2 = st.columns(2)
        col1.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.2f}")
        col1.metric("R² Score", f"{r2_score(y_test, y_pred):.2f}")
        col2.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

        # Feature Importance
        st.subheader("🔍 Feature Importance")
        importance = pd.DataFrame({
            'Feature': features,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        st.bar_chart(importance.set_index('Feature'))

        # Forecasting
        st.subheader(f"🔮 {forecast_days}-Day Price Forecast")
        
        # Create future data points
        last_data = X.iloc[-1:].copy()
        forecast_values = []
        
        for day in range(1, forecast_days+1):
            # Modify date-related features if they exist
            for col in features:
                if '_year' in col and (X[col].max() > X[col].min()):
                    last_data[col] = last_data[col] + day//365
                elif '_month' in col and (X[col].max() > X[col].min()):
                    last_data[col] = (last_data[col] + day//30) % 12 + 1
                elif '_day' in col and (X[col].max() > X[col].min()):
                    last_data[col] = (last_data[col] + day) % 31 + 1
            
            scaled_data = scaler.transform(last_data)
            forecast_values.append(model.predict(scaled_data)[0])

        # Plot forecast
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(range(len(y)), y, 'b-', label='Historical Data')
        ax.plot(range(len(y), len(y)+forecast_days), forecast_values, 'r--', label='Forecast')
        ax.axvline(x=len(y), color='k', linestyle='--')
        ax.set_title(f"Stock Price Forecast (Next {forecast_days} Days)")
        ax.set_xlabel("Time Period")
        ax.set_ylabel("Stock Price")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # Show forecast table
        forecast_df = pd.DataFrame({
            "Day": range(1, forecast_days+1),
            "Date": pd.date_range(start=datetime.today(), periods=forecast_days),
            "Predicted Price": forecast_values
        })
        st.dataframe(forecast_df.style.format({
            "Predicted Price": "{:.2f}",
            "Date": lambda x: x.strftime('%Y-%m-%d')
        }))

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
