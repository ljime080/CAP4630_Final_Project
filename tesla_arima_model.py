import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import itertools
import warnings

warnings.filterwarnings('ignore')


# Download Tesla stock data
def download_stock_data(ticker='TSLA', period='5y'):
    """
    Download stock data using Yahoo Finance API

    Parameters:
    ticker (str): Stock ticker symbol
    period (str): Time period to download (e.g., '1y', '5y', 'max')

    Returns:
    pandas.DataFrame: DataFrame containing stock data
    """
    print(f"Downloading {ticker} stock data for the past {period}...")
    stock_data = yf.download(ticker, period=period)
    return stock_data


# Check stationarity using Augmented Dickey-Fuller test
def check_stationarity(time_series):
    """
    Perform Augmented Dickey-Fuller test to check for stationarity

    Parameters:
    time_series (pandas.Series): Time series data to test

    Returns:
    bool: True if stationary, False otherwise
    """
    result = adfuller(time_series.dropna())
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')

    # If p-value is less than 0.05, the series is stationary
    return result[1] < 0.05


# Difference the series until it becomes stationary
def make_stationary(time_series):
    """
    Difference the time series until it becomes stationary

    Parameters:
    time_series (pandas.Series): Original time series

    Returns:
    tuple: (differenced_series, d_value)
    """
    differenced_series = time_series.copy()
    d_value = 0

    while not check_stationarity(differenced_series) and d_value < 2:
        differenced_series = differenced_series.diff().dropna()
        d_value += 1
        print(f"\nAfter differencing {d_value} time(s):")

    return differenced_series, d_value


# Identify optimal p and q values using ACF and PACF plots
def identify_p_q_values(differenced_series):
    """
    Create ACF and PACF plots to identify p and q values

    Parameters:
    differenced_series (pandas.Series): Stationary time series
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # ACF plot helps identify q value
    plot_acf(differenced_series, ax=ax1, lags=40)
    ax1.set_title('Autocorrelation Function (ACF)')

    # PACF plot helps identify p value
    plot_pacf(differenced_series, ax=ax2, lags=40)
    ax2.set_title('Partial Autocorrelation Function (PACF)')

    plt.tight_layout()
    plt.show()


# Find optimal ARIMA parameters using AIC
def find_optimal_arima_params(time_series, d_value, max_p=3, max_q=3):
    """
    Find optimal p and q values for ARIMA model using AIC

    Parameters:
    time_series (pandas.Series): Time series data
    d_value (int): Order of integration (d parameter)
    max_p (int): Maximum p value to test
    max_q (int): Maximum q value to test

    Returns:
    tuple: Optimal order (p, d, q)
    """
    best_aic = float('inf')
    best_order = None

    # Create a progress table
    print("\nFinding optimal ARIMA parameters:")
    print(f"{'p':<4}{'q':<4}{'AIC':<12}{'Status':<10}")
    print("-" * 30)

    for p in range(max_p + 1):
        for q in range(max_q + 1):
            # Skip if both p and q are 0
            if p == 0 and q == 0:
                continue

            try:
                model = ARIMA(time_series, order=(p, d_value, q))
                results = model.fit()
                aic = results.aic

                print(f"{p:<4}{q:<4}{aic:<12.4f}{'OK':<10}")

                if aic < best_aic:
                    best_aic = aic
                    best_order = (p, d_value, q)

            except Exception as e:
                print(f"{p:<4}{q:<4}{'N/A':<12}{'Failed':<10}")

    print(f"\nBest ARIMA order: {best_order} (AIC: {best_aic:.4f})")
    return best_order


# Build and train ARIMA model
def build_arima_model(time_series, order):
    """
    Build and fit ARIMA model

    Parameters:
    time_series (pandas.Series): Time series data
    order (tuple): ARIMA order (p, d, q)

    Returns:
    ARIMA model: Fitted ARIMA model
    """
    model = ARIMA(time_series, order=order)
    fitted_model = model.fit()
    print(fitted_model.summary())
    return fitted_model


# Forecast future values and plot results
def forecast_and_plot(model, time_series, steps=30):
    """
    Forecast future values and plot results

    Parameters:
    model: Fitted ARIMA model
    time_series (pandas.Series): Original time series
    steps (int): Number of steps to forecast
    """
    # Forecast
    forecast_result = model.get_forecast(steps=steps)
    forecast_mean = forecast_result.predicted_mean
    forecast_conf_int = forecast_result.conf_int(alpha=0.05)

    # Create a date range for the forecast
    last_date = time_series.index[-1]
    forecast_index = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                   periods=steps,
                                   freq=time_series.index.freq or 'B')

    # Assign the forecasted values to the date range
    forecast_series = pd.Series(forecast_mean.values, index=forecast_index)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(time_series.iloc[-365:], label='Historical Data (Last Year)')
    plt.plot(forecast_series, label='Forecast', color='red')
    plt.fill_between(forecast_index,
                     forecast_conf_int.iloc[:, 0],
                     forecast_conf_int.iloc[:, 1],
                     color='pink', alpha=0.3, label='95% Confidence Interval')
    plt.title('Tesla Stock Price Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Find optimal ARIMA parameters using grid search
def find_best_arima_model(time_series, max_p=45, max_d=2, max_q=45):
    """
    Use grid search to find the best ARIMA model

    Parameters:
    time_series (pandas.Series): Time series data
    max_p, max_d, max_q (int): Maximum values for p, d, q parameters

    Returns:
    tuple: Best ARIMA order (p,d,q)
    """
    print("\nPerforming grid search to find the best ARIMA model...")

    # Define the p, d and q parameters to take any value between
    p = range(0, max_p + 1)
    d = range(0, max_d + 1)
    q = range(0, max_q + 1)

    # Generate all different combinations of p, d and q
    pdq = list(itertools.product(p, d, q))

    # Remove combinations where p=0 and q=0 (not valid ARIMA models)
    pdq = [x for x in pdq if not (x[0] == 0 and x[2] == 0)]

    # Create table header
    print(f"{'p':<4}{'d':<4}{'q':<4}{'AIC':<12}{'BIC':<12}{'Status':<10}")
    print("-" * 46)

    best_aic = float('inf')
    best_order = None

    # Run a grid with all combinations
    for param in pdq:
        try:
            # Train the model
            model = ARIMA(time_series, order=param)
            results = model.fit()

            # Print results
            print(f"{param[0]:<4}{param[1]:<4}{param[2]:<4}{results.aic:<12.4f}{results.bic:<12.4f}{'OK':<10}")

            # Update if this is better than the best so far
            if results.aic < best_aic:
                best_aic = results.aic
                best_order = param

        except Exception as e:
            print(f"{param[0]:<4}{param[1]:<4}{param[2]:<4}{'N/A':<12}{'N/A':<12}{'Failed':<10}")

    print(f"\nBest ARIMA order: {best_order} (AIC: {best_aic:.4f})")
    return best_order


# Main function
def main():
    # Download Tesla stock data
    tesla_data = download_stock_data('TSLA', '5y')
    print(f"Data shape: {tesla_data.shape}")
    print(tesla_data.head())

    # Use adjusted close prices for the analysis
    closing_prices = tesla_data['Close']

    # Plot the original time series
    #plt.figure(figsize=(12, 6))
    closing_prices.plot()
    plt.title('Tesla Adjusted Close Price')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.grid(True)
    plt.show()

    # Make sure we have a valid frequency for forecasting
    if closing_prices.index.freq is None:
        closing_prices.index = pd.DatetimeIndex(closing_prices.index)
        closing_prices = closing_prices.asfreq('B', method='pad')  # Business day frequency
        print("Set time series frequency to business days")

    # Step 1: Visual analysis and determining d parameter
    print("\n=== Step 1: Determining differencing parameter (d) ===")

    # Check if the original series is stationary
    print("\nChecking stationarity of the original series:")
    original_stationary = check_stationarity(closing_prices)

    if not original_stationary:
        print("\nOriginal series is not stationary. Differencing to make it stationary...")
        differenced_series, d_value = make_stationary(closing_prices)
    else:
        print("\nOriginal series is stationary.")
        differenced_series, d_value = closing_prices, 0

    # Step 2: Show ACF/PACF plots for educational purposes
    print("\n=== Step 2: Visual inspection with ACF/PACF plots ===")
    print("\nShowing ACF and PACF plots for visual reference:")
    max_p, max_d, max_q = identify_p_q_values(differenced_series)

    # Step 3: Automated grid search for best parameters
    print("\n=== Step 3: Grid search for optimal p and q values ===")
    # Using d_value from ADF test, search for best p and q
    best_order = find_best_arima_model(closing_prices, max_p=max_p, max_d=max_d, max_q=max_q)

    print(f"\nBuilding ARIMA{best_order} model with automatically determined parameters:")
    arima_model = build_arima_model(closing_prices, order=best_order)

    # Forecast future values
    print("\nForecasting future values:")
    forecast_and_plot(arima_model, closing_prices)


if __name__ == "__main__":
    main()