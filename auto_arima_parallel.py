import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import warnings
import math
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA


from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

import concurrent.futures
from tqdm import tqdm

warnings.filterwarnings('ignore')


def get_data(ticker):
    df = yf.download(ticker, period='5y')['Close']
    # Adjusting for the yfinance output:
    # In this example, we assume the ticker is TSLA
    df = df['TSLA'].reset_index()
    df.columns = ['Date', 'Price']
    df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d")
    df = df.sort_values(by='Date')
    return df


def check_stationarity(time_series):
    result = adfuller(time_series.dropna())
    return result[1] < 0.05


def make_stationary(time_series):
    differenced_series = time_series.copy()
    d_value = 0
    # Difference until the series is stationary (up to 2 differences)
    while not check_stationarity(differenced_series) and d_value < 2:
        differenced_series = differenced_series.diff().dropna()
        d_value += 1
    return differenced_series, d_value


def get_ar_ma_parameters(ts_stationary):
    order_selection = sm.tsa.arma_order_select_ic(ts_stationary, max_ar=5, max_ma=5, ic=['aic', 'bic', 'hqic'])
    best_aic  = order_selection.aic_min_order
    best_bic  = order_selection.bic_min_order
    best_hqic = order_selection.hqic_min_order
    # Choose the order—if they agree, use that; otherwise, use BIC for simplicity.
    best_order = best_aic if (best_aic == best_bic == best_hqic) else best_bic
    p, q = best_order
    return p, q


def fit_arima_model(data, p, d, q):
    # Disable enforced stationarity and invertibility to avoid numerical issues
    model = ARIMA(data, order=(p, d, q), enforce_stationarity=False, enforce_invertibility=False)
    results = model.fit()
    return results


def get_yearly_splits(df, forecast_horizon=30):
    """
    Splits the data by calendar year. For each year, the training set is the data within that year,
    and the forecast (test) set is the next `forecast_horizon` rows after the last date of that year.
    """
    years = sorted(df['Date'].dt.year.unique())
    splits = []
    i = 0
    for year in years:
        # Training data: all rows for the given year
        train_data = df[df['Date'].dt.year == year].copy()
        last_date = train_data['Date'].max()
        # Forecast set: next forecast_horizon rows from the overall dataset (if available)
        test_data = df[df['Date'] > last_date].copy().head(forecast_horizon)
        if not test_data.empty:
            splits.append((i + 1, train_data, test_data))
            i +=1
    return splits


def get_metrics(model, test_set):
    forecasts = model.forecast(steps=len(test_set))
    mae = mean_absolute_error(test_set, forecasts)
    mse = mean_squared_error(test_set, forecasts)
    mape = mean_absolute_percentage_error(test_set, forecasts)
    return forecasts, mae, mse, mape



def process_year_split(train_data, test_data , train_year):
    """
    For one yearly split, makes the training data stationary,
    selects AR and MA parameters, fits an ARIMA model, and forecasts the test period.
    Returns a dictionary with the fitted model, forecast values, and confidence intervals.
    """
    # Make the training series stationary and determine the order of differencing (d)
    train_stationary, d = make_stationary(train_data.Price)
    # Determine AR and MA orders (p and q) from the stationary series
    p, q = get_ar_ma_parameters(train_stationary)
    # Fit the ARIMA model on the original (non-differenced) training data
    model_result = fit_arima_model(train_data.Price, p, d, q)
    # Forecast the test period; get_forecast provides confidence intervals
    forecast_cv, mae, mse, mape = get_metrics(model_result, test_data.Price)

    forecast_obj = model_result.get_forecast(steps=len(test_data))
    forecast_mean = forecast_obj.predicted_mean
    forecast_conf_int = forecast_obj.conf_int()

    return {
        'model': model_result,
        'model_orders': (p, d, q),
        'aic': model_result.aic,
        'forecast_mean': forecast_mean,
        'forecast_conf_int': forecast_conf_int,
        'train': train_data,
        'test': test_data,
        'mae': mae,
        'mse': mse,
        'mape': mape
    }



def plot_test_results(final_model, train, test, p, d, q):
    test_forecasts = final_model.get_forecast(steps=len(test.Price))
    test_forecast_mean = test_forecasts.predicted_mean
    test_forecast_confint = test_forecasts.conf_int()

    plt.figure(figsize=(12, 6))
    plt.plot(train.Date, train.Price, label='Training Data')
    plt.plot(test.Date, test.Price, label='True Test Data', color='green')
    plt.plot(test.Date, test_forecast_mean, label='Forecast', color='red')
    plt.fill_between(test.Date, 
                     test_forecast_confint.iloc[:, 0], 
                     test_forecast_confint.iloc[:, 1],
                     color='pink', alpha=0.3, label='Confidence Interval')
    plt.title(f"Test Set Forecast with Confidence Intervals using ARIMA ({p}, {d}, {q})")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()

    plt.savefig("./plots/test_results_trained_parallel.png", dpi=300)
    plt.show()


def process_split_wrapper(split):
    year, train_data, test_data = split
    res = process_year_split(train_data, test_data, year)
    res['year'] = year
    return res

def main():
    # Download 5 years of TSLA data
    df = get_data(ticker=['TSLA'])
    
    # Create yearly splits: each tuple is (year, train_data, test_data)
    splits = get_yearly_splits(df, forecast_horizon=30)
    
    # Process the yearly splits in parallel using ProcessPoolExecutor, with tqdm to show progress
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_split_wrapper, splits),
                            total=len(splits),
                            desc="Processing Yearly Splits"))
    
    # Create subplots: 3 rows x 2 columns for the CV results
    fig, axs = plt.subplots(3, 2, figsize=(30, 20))
    axs = axs.flatten()
    
    # Plot the forecast results for each split
    for i, res in enumerate(results):
        train_data = res['train']
        test_data = res['test']
        axs[i].plot(train_data.Date, train_data.Price, label='Training Data')
        axs[i].plot(test_data.Date, test_data.Price, label='True Test Data', color='green')
        axs[i].plot(test_data.Date, res['forecast_mean'], label='Forecast', color='red')
        axs[i].fill_between(test_data.Date, 
                            res['forecast_conf_int'].iloc[:, 0], 
                            res['forecast_conf_int'].iloc[:, 1],
                            color='pink', alpha=0.3, label='Confidence Interval')
        axs[i].set_title(f"Year {res['year']}: Forecast for Next 30 Days MAPE = {res['mape']:.2f}")
        axs[i].set_xlabel("Date")
        axs[i].set_ylabel("Price")
        axs[i].legend()
    
    # Turn off any unused subplots if the number of splits is less than 6
    for j in range(len(results), len(axs)):
        axs[j].axis('off')
    
    fig.savefig("./plots/cv_results_trained_parallel.png", dpi=300)
    plt.show()
    
    # Choose the best model based on minimum AIC across CV splits
    best_cv_result = min(results, key=lambda x: x['aic'])
    best_p, best_d, best_q = best_cv_result['model_orders']
    
    # Use the last split for final model fitting
    year, train_data, test_data = splits[-1]
    final_arima = fit_arima_model(train_data.Price, best_p, best_d, best_q)
    plot_test_results(final_arima, train_data, test_data, best_p, best_d, best_q)
    
    final_arima.save('./models/arima_model_trained_parallel.pkl')



if __name__ == "__main__":
    main()


