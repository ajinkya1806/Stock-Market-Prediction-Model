import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set page configuration
st.set_page_config(page_title="Stock Price Prediction", layout="wide")

# Title
st.title("Stock Price Prediction with RNN Models")

# Dictionary to map company names to their model and dataset paths
companies = {
    "Bharti Airtel": {"model": "/media/ajk1806/AAKHRI MAUKA/InternSHip/ATTEMPT_001 (copy)/models/Airtel_model.h5", "data": "/media/ajk1806/AAKHRI MAUKA/InternSHip/ATTEMPT_001 (copy)/data/BHARTIARTL.csv"},
    "Reliance Industries": {"model": "/media/ajk1806/AAKHRI MAUKA/InternSHip/ATTEMPT_001 (copy)/models/RELIANCE_model.h5", "data": "/media/ajk1806/AAKHRI MAUKA/InternSHip/ATTEMPT_001 (copy)/data/RELIANCE.csv"},
    "TCS": {"model": "/media/ajk1806/AAKHRI MAUKA/InternSHip/ATTEMPT_001 (copy)/models/TCS_model.h5", "data": "/media/ajk1806/AAKHRI MAUKA/InternSHip/ATTEMPT_001 (copy)/data/TCS.csv"},
    "ITC": {"model": "/media/ajk1806/AAKHRI MAUKA/InternSHip/ATTEMPT_001 (copy)/models/rnn_model.h5", "data": "/media/ajk1806/AAKHRI MAUKA/InternSHip/ATTEMPT_001 (copy)/data/ITC.csv"},
    "HDFC Bank": {"model": "/media/ajk1806/AAKHRI MAUKA/InternSHip/ATTEMPT_001 (copy)/models/HDFC_model.h5", "data": "/media/ajk1806/AAKHRI MAUKA/InternSHip/ATTEMPT_001 (copy)/data/HDFCBANK.csv"},
}

# Sidebar for company selection
st.sidebar.header("Select Company")
company = st.sidebar.selectbox("Choose a company", list(companies.keys()))

# Load dataset
@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    return df

# Load model
@st.cache_resource
def load_company_model(model_path):
    return load_model(model_path)

# Function to create sequences
def create_sequences(data, seq_length=60):
    X = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
    return np.array(X)

# Load data and model for selected company
df = load_data(companies[company]["data"])
model = load_company_model(companies[company]["model"])

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df[['Close']].values)

# Prepare the most recent sequence for prediction
recent_sequence = scaled_data[-60:]  # Last 60 days
X_input = np.reshape(recent_sequence, (1, 60, 1))

# Make prediction
predicted_scaled = model.predict(X_input)
predicted_price = scaler.inverse_transform(predicted_scaled)[0][0]

# Display prediction
st.subheader(f"Predicted Stock Price for {company}")
st.write(f"The predicted closing price for the next day is: **₹{predicted_price:.2f}**")

# Train-test split for evaluation and plotting
seq_length = 60
X, y = [], []
for i in range(len(scaled_data) - seq_length):
    X.append(scaled_data[i:i + seq_length, 0])
    y.append(scaled_data[i + seq_length, 0])
X = np.array(X)
y = np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Split data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Evaluate model on test set
predicted_scaled_test = model.predict(X_test)
predicted_test = scaler.inverse_transform(predicted_scaled_test)
actual_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Calculate metrics
mse = mean_squared_error(actual_test, predicted_test)
mae = mean_absolute_error(actual_test, predicted_test)
r2 = r2_score(actual_test, predicted_test)

# Display metrics
st.subheader("Model Performance Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Mean Squared Error (MSE)", f"{mse:.4f}")
col2.metric("Mean Absolute Error (MAE)", f"{mae:.4f}")
col3.metric("R² Score", f"{r2:.4f}")

# Plot 1: Actual vs Predicted Prices
st.subheader("Actual vs Predicted Prices")
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=np.arange(len(actual_test)), y=actual_test.flatten(), mode='lines', name='Actual', line=dict(color='red')))
fig1.add_trace(go.Scatter(x=np.arange(len(predicted_test)), y=predicted_test.flatten(), mode='lines', name='Predicted'))
fig1.update_layout(title="Actual vs Predicted Stock Prices", xaxis_title="Test Sample", yaxis_title="Price (₹)")
st.plotly_chart(fig1, use_container_width=True)

# Plot 2: Historical Stock Price Trend
st.subheader("Historical Stock Price Trend")
fig2 = px.line(df, x='Date', y='Close', title=f"{company} Historical Closing Prices")
fig2.update_layout(xaxis_title="Date", yaxis_title="Price (₹)")
st.plotly_chart(fig2, use_container_width=True)

# Plot 3: Volume Trend
st.subheader("Trading Volume Trend")
fig3 = px.line(df, x='Date', y='Volume', title=f"{company} Trading Volume Over Time")
fig3.update_layout(xaxis_title="Date", yaxis_title="Volume")
st.plotly_chart(fig3, use_container_width=True)

# Optional: Allow users to input custom sequence (advanced feature)
st.subheader("Custom Prediction (Optional)")
with st.expander("Enter Custom 60-Day Closing Prices"):
    custom_input = st.text_area("Enter 60 comma-separated closing prices (e.g., 100.5, 101.2, ...)")
    if st.button("Predict with Custom Input"):
        try:
            custom_prices = [float(x) for x in custom_input.split(",")]
            if len(custom_prices) == 60:
                custom_scaled = scaler.transform(np.array(custom_prices).reshape(-1, 1))
                custom_X = np.reshape(custom_scaled, (1, 60, 1))
                custom_pred_scaled = model.predict(custom_X)
                custom_pred = scaler.inverse_transform(custom_pred_scaled)[0][0]
                st.success(f"Predicted price with custom input: **₹{custom_pred:.2f}**")
            else:
                st.error("Please enter exactly 60 values.")
        except:
            st.error("Invalid input. Ensure you enter valid numbers separated by commas.")

# Footer
st.markdown("---")
st.write("Built with Streamlit and TensorFlow | Stock Prediction RNN Models")