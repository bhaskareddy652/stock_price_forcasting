import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.title("📈 30-Day Stock Price Forecasting App")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.write(df.head())

    # Feature and target selection
    features = ['nasdaq_index', 'sp500_index', 'inflation_rate', 'unemployment_rate', 'interest_rate', 'market_sentiment']
    target = 'stock_price'

    if all(col in df.columns for col in features + [target]):
        X = df[features]
        y = df[target]

        # Split and scale
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Random Forest Regressor with hyperparameter tuning
        rf = RandomForestRegressor(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
        }
        grid_search = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1, scoring='r2')
        grid_search.fit(X_train, y_train)

        best_rf = grid_search.best_estimator_
        y_pred = best_rf.predict(X_test)

        # Evaluation
        st.subheader("📈 Model Performance")
        st.write(f"**Best Parameters:** {grid_search.best_params_}")
        st.write(f"**MAE:** {mean_absolute_error(y_test, y_pred):.2f}")
        st.write(f"**MSE:** {mean_squared_error(y_test, y_pred):.2f}")
        st.write(f"**RMSE:** {np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")
        st.write(f"**R² Score:** {r2_score(y_test, y_pred):.2f}")

        # Forecast next 30 days
        st.subheader("🔮 30-Day Forecast")

        future_data = pd.DataFrame([X.mean()] * 30, columns=X.columns)
        future_scaled = scaler.transform(future_data)
        future_preds = best_rf.predict(future_scaled)

        # Plot
        fig, ax = plt.subplots()
        ax.plot(range(1, 31), future_preds, marker='o', linestyle='-', color='blue')
        ax.set_title('30-Day Stock Price Forecast')
        ax.set_xlabel('Day')
        ax.set_ylabel('Predicted Stock Price')
        st.pyplot(fig)

    else:
        st.error("❌ Your dataset is missing required columns.")
