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
            model = RandomForestRegressor
