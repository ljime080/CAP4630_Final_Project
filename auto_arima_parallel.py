import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import itertools
import warnings
import math
import concurrent.futures
from tqdm import tqdm

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

warnings.filterwarnings('ignore')


def get_data(ticker):
    df = yf.download(ticker, period='5y')['Close']
    # Adjusting for the yfinance output:
    # In this example, we assume the ticker is TSLA
    df = df['TSLA'].reset_index()
    df.columns = ['Date', 'Price']
    df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d")
    df = df.sort_values(by='Date')

    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[ (df.shape[0] - 30 ) :]
    train = train.sort_values(by='Date')
    test = test.sort_values(by='Date')
    return train, test


# Step 1: Check stationarity and difference until stationary
def check_stationarity(time_series):
    result = adfuller(time_series.dropna())
    #print('ADF Statistic:', result[0])
    #print('p-value:', result[1])
    return result[1] < 0.05


def make_stationary(time_series):
    differenced_series = time_series.copy()
    d_value = 0
    while not check_stationarity(differenced_series) and d_value < 2:
        differenced_series = differenced_series.diff().dropna()
        d_value += 1
        #print(f"\nAfter differencing {d_value} time(s):")
    return differenced_series, d_value


# Step 2: Get AR and MA parameters from stationary series
def get_ar_ma_parameters(ts_stationary):
    order_selection = sm.tsa.arma_order_select_ic(ts_stationary, max_ar=5, max_ma=5,  ic=['aic', 'bic', 'hqic'])

    best_aic  = order_selection.aic_min_order
    best_bic  = order_selection.bic_min_order
    best_hqic = order_selection.hqic_min_order

    # If all criteria agree, use that order.
    if best_aic == best_bic == best_hqic:
        best_order = best_aic
    else:
        # If they differ, you can:
        # Option 1: Pick the order from the criterion that best suits your priorities.
        #         For instance, BIC tends to favor simpler models.
        best_order = best_bic

    p, q = best_order
    return p, q


# Step 3: Fit the ARIMA model
def fit_arima_model(cv_train, p, d, q):
    model = ARIMA(cv_train, order=(p, d, q) , enforce_stationarity=False, enforce_invertibility=False)
    model = model.fit()
    return model


# Step 4: Forecast and calculate error metrics
def get_metrics(model, test_set):
    forecasts = model.forecast(steps=len(test_set))
    mae = mean_absolute_error(test_set, forecasts)
    mse = mean_squared_error(test_set, forecasts)
    mape = mean_absolute_percentage_error(test_set, forecasts)
    return forecasts, mae, mse, mape


# This function processes one CV fold; it is designed for parallel execution.
def process_fold(start, train, train_size, test_size):
    # Define training and test splits for this fold.
    cv_train = train.iloc[start: start + train_size]
    cv_test = train.iloc[start + train_size: start + train_size + test_size]
    
    # Print progress information for this fold.
    #print(f"Processing fold: Training indices {cv_train.index[0]} to {cv_train.index[-1]}, "
    #      f"Test indices {cv_test.index[0]} to {cv_test.index[-1]}")
    
    # Make series stationary and find d
    cv_train_stationary, d = make_stationary(cv_train.Price)
    # Get AR and MA parameters
    p, q = get_ar_ma_parameters(cv_train_stationary)
    # Fit the ARIMA model
    cv_model = fit_arima_model(cv_train.Price, p, d, q)
    # Forecast and calculate metrics
    forecast_cv, mae, mse, mape = get_metrics(cv_model, cv_test.Price)
    
    return {
        'model': cv_model,
        'model_orders': (p, d, q),
        'aic': cv_model.aic,
        'cv_test': cv_test.Price,
        'cv_forecasts': forecast_cv,
        'mae': mae,
        'mse': mse,
        'mape': mape
    }


def run_fold(args):
    return process_fold(*args)



def plot_cv_results(cv_results , n_rows, n_cols , n_splits):
    cv_fig, cv_axs = plt.subplots(n_rows, n_cols, figsize=(60, 30))
    # If more than one subplot, flatten the axes array.
    if n_splits > 1:
        axes = cv_axs.flatten()
    else:
        axes = [cv_axs]
    
    k = 0
    for i in range(n_rows):
        for j in range(n_cols):
            if k < len(cv_results):
                curr_result = cv_results[k]
                curr_cv_test = curr_result['cv_test']
                curr_forecast_cv = curr_result['cv_forecasts']
                curr_mse = curr_result['mse']
                curr_mae = curr_result['mae']
                curr_mape = curr_result['mape']
                p, d, q = curr_result['model_orders']
    
                plot_title = (f"p = {p}, d = {d}, q = {q}\n"
                              f"MSE = {curr_mse:.2f}, MAE = {curr_mae:.2f}, MAPE = {curr_mape:.2f}")
    
                axes[k].plot(curr_cv_test.index, curr_cv_test, label='CV test data', color='blue')
                axes[k].plot(curr_forecast_cv.index, curr_forecast_cv, label='CV forecast', color='orange')
                k += 1
            else:
                # Turn off any extra subplots.
                cv_axs[i, j].axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_test_results(final_model , train, test , p , d , q):
    test_forecasts = final_model.get_forecast(n = len(test) )
    test_forecast_mean = test_forecasts.mean
    test_forecast_confint = test_forecasts.conf_int()

    plt.figure(figsize=(12, 6))
    plt.plot(train.Date, train.Price, label='Training Data')
    plt.plot(test.Date, test.Price, label='True Test Data', color='green')
    plt.plot(test.Date, test_forecast_mean, label='Forecast', color='red')
    plt.fill_between(test.Date, 
                     test_forecast_confint.iloc[:, 0], 
                     test_forecast_confint.iloc[:, 1],
                     color='pink', alpha=0.3, label='Confidence Interval')
    plt.title("Test Set Forecast with Confidence Intervals using ARIMA ({p} , {d} , {q})")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()





def main():
    train, test = get_data(ticker=['TSLA'])
    
    # Fixed window sizes:
    train_size = 365
    test_size = 30
    window_size = train_size + test_size
    
    # Calculate number of CV splits
    n_splits = len(train) - window_size + 30
    print("Number of rolling CV splits:", n_splits)
    
    # Determine grid layout for subplots based on n_splits.
    n_cols = math.ceil(math.sqrt(n_splits))
    n_rows = math.ceil(n_splits / n_cols)
    print(f"Using a grid of {n_rows} rows and {n_cols} columns for the subplots.")
    
    # Prepare the list of fold arguments.
    fold_args = [(start, train, train_size, test_size) for start in range(n_splits)]
    
    # Parallel processing of the rolling CV folds.
    print("Starting parallel rolling cross validation using ProcessPoolExecutor...")
    cv_results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for result in tqdm(executor.map(run_fold, fold_args), total=n_splits, desc="Parallel CV"):
            cv_results.append(result)
    
    # Plot the results in a grid of subplots.
    plot_cv_results(cv_results , n_rows, n_cols , n_splits)

    print("fitting final model")
    best_cv_results = min(cv_results , key = lambda x: x['aic'] )
    best_p , best_d , best_q = best_cv_results['model_orders']

    final_arima = ARIMA(train, order=(best_p, best_d, best_q) , enforce_stationarity=False, enforce_invertibility=False )
    final_arima = final_arima.fit()

    plot_test_results(final_arima , train , test , best_p , best_d , best_q)
    


    final_arima.save('../models/arima_model.pkl')



if __name__ == "__main__":
    main()

