import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.title("📈 30-Day Stock Price Forecasting App")

# Sidebar for configuration
st.sidebar.header("Settings")
test_size = st.sidebar.slider("Test Size Ratio", 0.1, 0.5, 0.2, 0.05)
n_days = st.sidebar.number_input("Forecast Days", 30, 90, 30)

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        
        st.subheader("📊 Dataset Preview")
        st.write(df.head())

        # Let user select features and target
        all_columns = df.columns.tolist()
        features = st.multiselect(
            "Select features", 
            all_columns, 
            default=all_columns[:-1]  # Default to all columns except last
        )
        target = st.selectbox(
            "Select target variable", 
            all_columns, 
            index=len(all_columns)-1
        )

        if not features:
            st.error("❌ Please select at least one feature")
            st.stop()

        if len(df) < 30:
            st.error("❌ Dataset needs at least 30 observations")
            st.stop()

        # Prepare data
        X = df[features]
        y = df[target]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=42
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        with st.spinner('Training model...'):
            try:
                model = RandomForestRegressor(random_state=42)
                param_grid = {
                    'n_estimators': [50, 100, 150],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                }
                grid_search = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
                grid_search.fit(X_train_scaled, y_train)
                best_model = grid_search.best_estimator_
            except Exception as e:
                st.error(f"❌ Model training failed: {str(e)}")
                st.stop()

        # Evaluate
        y_pred = best_model.predict(X_test_scaled)
        
        st.subheader("📈 Model Performance")
        st.write(f"**Best Parameters:** {grid_search.best_params_}")
        
        col1, col2 = st.columns(2)
        col1.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.2f}")
        col1.metric("R² Score", f"{r2_score(y_test, y_pred):.2f}")
        col2.metric("RMSE", f"{np.sqrt(mean_squared_error(y_test, y_pred)):.2f}")

        # Forecast
        st.subheader(f"🔮 {n_days}-Day Forecast")
        last_values = X.iloc[-1:].values
        future_data = pd.DataFrame(
            np.repeat(last_values, n_days, axis=0), 
            columns=X.columns
        )
        future_scaled = scaler.transform(future_data)
        forecast = best_model.predict(future_scaled)

        # Plot forecast
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(range(1, n_days+1), forecast, 'b-', marker='o')
        ax.set_title(f"{n_days}-Day Price Forecast")
        ax.set_xlabel("Day")
        ax.set_ylabel("Predicted Price")
        ax.grid(True)
        st.pyplot(fig)

        # Show forecast table
        forecast_df = pd.DataFrame({
            "Day": range(1, n_days+1),
            "Predicted Price": forecast
        })
        st.dataframe(forecast_df.style.format({"Predicted Price": "{:.2f}"}))

    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
