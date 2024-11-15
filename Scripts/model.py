import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ARIMA Model
def arima(train, test):
    # Train ARIMA model
    model = ARIMA(train, order=(5, 1, 0))  # Adjust (p, d, q) as needed
    arima_model = model.fit()

    # Forecasting
    forecast = arima_model.forecast(steps=len(test))
    forecast.to_csv("Arima_forecast.csv")

    # Plot and evaluate
    plt.figure(figsize=(10, 6))
    plt.plot(test, label="Actual")
    plt.plot(forecast, label="ARIMA Forecast")
    plt.legend()
    plt.show()

    mae = mean_absolute_error(test, forecast)
    rmse = np.sqrt(mean_squared_error(test, forecast))
    mape = np.mean(np.abs((test - forecast) / test)) * 100

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")

# SARIMA Model
def sarima(train, test):
    # Train SARIMA model
    sarima_model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)).fit()

    # Forecasting
    sarima_forecast = sarima_model.forecast(steps=len(test))
    sarima_forecast.to_csv("Sarima_forecast.csv")

    # Plot and evaluate
    plt.figure(figsize=(10, 6))
    plt.plot(test, label="Actual")
    plt.plot(sarima_forecast, label="SARIMA Forecast")
    plt.legend()
    plt.show()

    mae = mean_absolute_error(test, sarima_forecast)
    rmse = np.sqrt(mean_squared_error(test, sarima_forecast))
    mape = np.mean(np.abs((test - sarima_forecast) / test)) * 100

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")

# LSTM Model
def lstm(train, test):
    # Scale data
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(np.array(train).reshape(-1, 1))
    test_scaled = scaler.transform(np.array(test).reshape(-1, 1))

    # Prepare data for LSTM
    def create_sequences(data, seq_length=60):
        X, y = [], []
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length, 0])
            y.append(data[i + seq_length, 0])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(train_scaled)
    X_train = np.expand_dims(X_train, axis=2)  # Add third dimension for LSTM

    # Define LSTM model
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train model
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=2)

    # Forecast
    X_test, _ = create_sequences(test_scaled)
    X_test = np.expand_dims(X_test, axis=2)
    lstm_forecast = model.predict(X_test)
    lstm_forecast = scaler.inverse_transform(lstm_forecast)
    
    # Save forecast as DataFrame
    lstm_forecast_df = pd.DataFrame(lstm_forecast, index=test.index[-len(lstm_forecast):], columns=["LSTM_Forecast"])
    lstm_forecast_df.to_csv("Lstm_forecast.csv")
    
    # Visualization
    plt.figure(figsize=(14, 7))
    plt.plot(test, label="Actual", color="blue")
    plt.plot(lstm_forecast_df.index, lstm_forecast_df["LSTM_Forecast"], label="LSTM Forecast", color="orange")
    plt.title("LSTM Stock Price Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

    # Evaluation
    aligned_test = test.iloc[-len(lstm_forecast):]
    mae = mean_absolute_error(aligned_test, lstm_forecast_df)
    rmse = np.sqrt(mean_squared_error(aligned_test, lstm_forecast_df))
    mape = np.mean(np.abs((aligned_test.values - lstm_forecast) / aligned_test.values)) * 100

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAPE: {mape:.2f}%")
