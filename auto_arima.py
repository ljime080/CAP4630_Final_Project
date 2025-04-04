import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import itertools
import warnings
import math

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.model_selection import TimeSeriesSplit

from sklearn.metrics import mean_absolute_error, mean_squared_error , mean_absolute_percentage_error


warnings.filterwarnings('ignore')




def get_data(ticker):

    df = yf.download(ticker, period = '1y' )['Close'] 
    df.columns.name = None 
    df = df['TSLA'].reset_index()
    df.columns = ['Date' , 'Price']
    df['Date'] = pd.to_datetime(df['Date'] , format= "%Y-%m-%d")
    df.head()


    train_size = int(len(df) * 0.8)
    train, test = df.iloc[:train_size], df.iloc[train_size:]
    train = train.sort_values(by = 'Date')
    test = test.sort_values(by = 'Date')

    return train , test


# Step 1 
# create function that returns a time series stationary, with the value of d

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
    

# Step2 
# create function that returns values of p and q given stationary time series
def get_ar_ma_parameters(ts_stationary):
    order_selection = sm.tsa.arma_order_select_ic(ts_stationary, max_ar=5, max_ma=5, ic=['aic'])
    p, q = order_selection.aic_min_order

    return p, q


# Step3
# create function that fits the arima and return the model  
def fit_arima_model(cv_train , p,d,q):
    model = ARIMA(cv_train , order = (p,d,q) )
    model =  model.fit()


    return model

# Step 4
# create function that calculates the metrics (mae, mse, mape)
def get_metrics(model , test_set):
    # Forecast for the cv_test horizon
    #print(f"length of cv_test: {len(test_set)}")
    forecasts = model.forecast( steps = len(test_set) )
    
    #print(f"length of forecast: {len(forecasts)}")

    mae = mean_absolute_error(test_set , forecasts )
    mse = mean_squared_error(test_set , forecasts )
    mape = mean_absolute_percentage_error(test_set , forecasts )

    return forecasts , mae, mse, mape
    

# Step 4
# get the parameters of the model with best results

# Step 5
# Fit model with best parameters on train data

# Step 6
# Predict testing data and save model


def main():
    train, test = get_data(ticker=['TSLA'])
    
    # Fixed window sizes:
    train_size = 30
    test_size = 30
    window_size = train_size + test_size

    # For rolling window, allow overlapping folds with step size = 1.
    n_splits = len(train) - window_size + 1
    print("Number of rolling CV splits:", n_splits)

    # Determine grid layout for subplots based on n_splits.
    n_cols = math.ceil(math.sqrt(n_splits))
    n_rows = math.ceil(n_splits / n_cols)
    print(f"Using a grid of {n_rows} rows and {n_cols} columns for the subplots.")

    cv_fig, cv_axs = plt.subplots(n_rows, n_cols, figsize=(60, 30))
    # If there is more than one subplot, flatten for easy indexing.
    if n_splits > 1:
        axes = cv_axs.flatten()
    else:
        axes = [cv_axs]

    cv_results = []
    fold = 1

    print("Starting rolling cross validation with fixed windows (30 train / 30 test)")
    # Rolling window: slide one observation at a time.
    for start in range(0, n_splits):
        # Define training and test splits for this fold.
        cv_train = train.iloc[start : start + train_size]
        cv_test  = train.iloc[start + train_size : start + train_size + test_size]
        
        print(f"\nFold {fold}:")
        print("Training indices:", cv_train.index[0], "to", cv_train.index[-1])
        print("Test indices:", cv_test.index[0], "to", cv_test.index[-1])

        print("Making time series stationary and finding d")
        cv_train_stationary, d = make_stationary(cv_train.Price)

        print("Getting p and q parameters")
        p, q = get_ar_ma_parameters(cv_train_stationary)

        # Train auto ARIMA on the training portion of the fold
        print("Fitting model")
        cv_model = fit_arima_model(cv_train.Price, p, d, q)
        
        # Forecast and calculate metrics
        print("Forecasting and calculating")
        forecast_cv, mae, mse, mape = get_metrics(cv_model, cv_test.Price)
        
        temp_results = {
            'model': cv_model,
            'model_orders': (p, d, q),
            'cv_test': cv_test.Price,
            'cv_forecasts': forecast_cv,
            'mae': mae,
            'mse': mse,
            'mape': mape
        }
        cv_results.append(temp_results)
        fold += 1

    # Plot the results in a grid of subplots
    k = 0
    for i in range(n_rows):         # Loop over rows
        for j in range(n_cols):     # Loop over columns
            if k < len(cv_results):
                curr_model = cv_results[k]['model']
                curr_cv_test = cv_results[k]['cv_test']
                curr_forecast_cv = cv_results[k]['cv_forecasts']
                curr_mse = cv_results[k]['mse']
                curr_mae = cv_results[k]['mae']
                curr_mape = cv_results[k]['mape']
                p, d, q = cv_results[k]['model_orders']

                plot_title = (f"p = {p}, d = {d}, q = {q}\n"
                              f"MSE = {curr_mse:.2f}, MAE = {curr_mae:.2f}, MAPE = {curr_mape:.2f}")
                
                cv_axs[i, j].plot(curr_cv_test.index, curr_cv_test, label='CV test data', color='blue')
                cv_axs[i, j].plot(curr_forecast_cv.index, curr_forecast_cv, label='CV forecast', color='orange')
                cv_axs[i, j].set_title(plot_title)
                cv_axs[i, j].legend()
                k += 1
            else:
                # Turn off any extra subplots
                cv_axs[i, j].axis('off')

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    main()



