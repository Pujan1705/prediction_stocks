import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from datetime import datetime
import yfinance as yf
import matplotlib.pyplot as plt
import tempfile


def store_prediction(date, stock_name, actual_price, predicted_price):
    try:
        conn = sqlite3.connect('predictions2.db')
        c = conn.cursor()
        c.execute("CREATE TABLE IF NOT EXISTS predictions (date DATE, stock_name TEXT, actual_price REAL, predicted_price REAL, price_diff REAL, percent_diff REAL)")
        price_diff = predicted_price - actual_price
        percent_diff = 100 - (predicted_price / actual_price) * 100
        c.execute("INSERT INTO predictions (date, stock_name, actual_price, predicted_price, price_diff, percent_diff) VALUES (?, ?, ?, ?, ?, ?)",
                  (date, stock_name, actual_price, predicted_price, price_diff, percent_diff))
        conn.commit()
        conn.close()
    except Exception as e:
        st.error(f"Error storing prediction in the database: {e}")


def main():
    st.title("Stock Price Prediction")

    # Get the company symbol from the user
    stock_symbol = st.text_input("Enter the company symbol (e.g., AAPL):")

    # Fetch historical stock data using yfinance
    try:
        stock_data = yf.download(stock_symbol, start='2000-01-01', end='2023-06-06')
        stock_data.reset_index(inplace=True)
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return

    st.title(f"Stock Price Prediction for {stock_symbol}")

    # Convert date column to datetime
    stock_data['Date'] = pd.to_datetime(stock_data['Date'])

    # Add columns for year, month, and day
    stock_data['Year'] = stock_data['Date'].dt.year
    stock_data['Month'] = stock_data['Date'].dt.month
    stock_data['Day'] = stock_data['Date'].dt.day

    # Set the input date
    input_date_str = st.date_input("Enter the date for which you want to predict the closing price:")
    input_date = datetime.combine(input_date_str, datetime.min.time())

    # Filter the data for dates before the input date
    data_before_date = stock_data[stock_data['Date'] < input_date]

    # Set the number of days to predict ahead
    days_to_predict = 1

    # Shift the target variable (Closing Price) up by the number of days to predict
    data_before_date['Close_Target'] = data_before_date['Close'].shift(-days_to_predict)

    # Remove the last row, which will have a NaN target value
    data_before_date.dropna(inplace=True)

    # Define the features and target variable
    features = ['High', 'Low', 'Open', 'Adj Close', 'Volume']  # Include 'Open' as a feature
    target = 'Close_Target'

    # Split the data into training and test sets
    if len(data_before_date) > 1:
        X_train, X_test, y_train, y_test = train_test_split(
            data_before_date[features], data_before_date[target], test_size=0.2, random_state=0
        )
    else:
        X_train, X_test, y_train, y_test = [], [], [], []

    # Check if there are samples in the training and test sets
    if len(X_train) > 0 and len(X_test) > 0:
        # Fit a random forest regressor
        model = RandomForestRegressor(n_estimators=100, random_state=0)
        model.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = model.predict(X_test)

        # Calculate mean absolute error on the test set
        mae = mean_absolute_error(y_test, y_pred)

        # Predict the closing price for the input date
        input_features = np.array(data_before_date.iloc[-1][features]).reshape(1, -1)
        predicted_close = model.predict(input_features)[0]
        predicted_close_with_mae = predicted_close + mae

        # Store the actual closing price and predicted price in the database
        actual_close = data_before_date.iloc[-1]['Close']
        store_prediction(input_date.date(), stock_symbol, actual_close, predicted_close_with_mae)

        # Display the results
        price_diff = predicted_close - actual_close
        percent_diff = 100 - (predicted_close / actual_close) * 100
        st.write("Mean Absolute Error:", mae)
        st.write("Predicted closing price for", input_date.date(), "is", predicted_close)
        st.write("Predicted closing price with MAE for", input_date.date(), "is", predicted_close_with_mae)
        st.write("Actual closing price for", input_date.date(), "is", actual_close)
        st.write("Price difference:", price_diff)
        st.write("Percentage difference:", percent_diff)

        # Plot historical stock prices along with predicted closing price
        plt.figure(figsize=(12, 6))
        plt.plot(stock_data['Date'], stock_data['Close'], label='Historical Close Price')
        plt.axvline(x=input_date, color='r', linestyle='--', label='Prediction Date')
        plt.plot(input_date, predicted_close, marker='o', markersize=8, color='g', label='Predicted Close Price')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        plt.title(f'{stock_symbol} - Historical Close Prices and Prediction')
        plt.legend()
        st.pyplot(plt)

        # Download the database file
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.close()
        conn = sqlite3.connect(temp_file.name)
        c = conn.cursor()
        c.execute("ATTACH DATABASE 'predictions.db' AS original")
        c.execute("CREATE TABLE predictions AS SELECT * FROM original.predictions")
        conn.commit()
        conn.close()
        st.markdown(
            f'<a href="{temp_file.name}" download>Click here to download the database file</a>',
            unsafe_allow_html=True
        )
    else:
        st.warning("Not enough samples in the dataset for training and testing")


if __name__ == "__main__":
    main()
