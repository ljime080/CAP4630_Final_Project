
# Import all necessary packages.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf


from tensorflow.keras.layers import Input, LSTM, GRU, Dense, RepeatVector, TimeDistributed, Layer, Bidirectional, MultiHeadAttention, Dropout, LayerNormalization, Embedding, GlobalAveragePooling1D
from keras.saving import register_keras_serializable
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

import os
import random
import warnings
import streamlit as st


warnings.filterwarnings('ignore')

os.environ['PYTHONHASHSEED'] = '42'
os.environ['TF_DETERMINISTIC_OPS'] = '1' 


np.random.seed(42)
tf.random.set_seed(42)




@register_keras_serializable()
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        # Create trainable weights for the attention mechanism.
        self.W = self.add_weight(name="att_weight",
                                 shape=(input_shape[-1], input_shape[-1]),
                                 initializer="random_normal",
                                 trainable=True)
        self.b = self.add_weight(name="att_bias",
                                 shape=(input_shape[-1],),
                                 initializer="zeros",
                                 trainable=True)
        self.V = self.add_weight(name="att_var",
                                 shape=(input_shape[-1], 1),
                                 initializer="random_normal",
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, inputs):
        score = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        attention_weights = tf.nn.softmax(tf.tensordot(score, self.V, axes=1), axis=1)
        context_vector = attention_weights * inputs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector



def build_lstm_bidirectional_seq2seq_model(window_size, forecast_horizon, feature_count=1):
    # Encoder (bidirectional LSTM)
    encoder_inputs = Input(shape=(window_size, feature_count))
    encoder = Bidirectional(LSTM(128, return_state=True))
    encoder_outputs = encoder(encoder_inputs)
    # The bidirectional LSTM returns: outputs, forward_h, forward_c, backward_h, backward_c
    _, forward_h, forward_c, backward_h, backward_c = encoder_outputs
    state_h = Dense(128, activation='tanh')(((forward_h + backward_h) / 2))
    state_c = Dense(128, activation='tanh')(((forward_c + backward_c) / 2))
    
    # Decoder
    decoder_inputs = Input(shape=(forecast_horizon, feature_count))
    decoder_lstm = LSTM(128, return_sequences=True)
    decoder_outputs = decoder_lstm(decoder_inputs, initial_state=[state_h, state_c])
    decoder_dense = Dense(feature_count)
    outputs = decoder_dense(decoder_outputs)
    
    model = Model([encoder_inputs, decoder_inputs], outputs)
    model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')
    model.summary()
    return model

def build_lstm_seq2seq_model(window_size, forecast_horizon):
    # Single-input variant that outputs a sequence forecast.
    encoder_inputs = Input(shape=(window_size, 1))
    encoder_lstm = LSTM(64, return_state=True)
    encoder_output, state_h, state_c = encoder_lstm(encoder_inputs)
    encoder_states = [state_h, state_c]
    
    decoder_inputs = RepeatVector(forecast_horizon)(encoder_output)
    decoder_lstm = LSTM(64, return_sequences=True)
    decoder_outputs = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_outputs = TimeDistributed(Dense(1))(decoder_outputs)
    
    model = Model(encoder_inputs, decoder_outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    model.summary()
    return model

def build_gru_seq2seq_model(window_size, forecast_horizon, latent_dim=64):
    # GRU-based Seq2Seq model.
    encoder_inputs = Input(shape=(window_size, 1))
    encoder_gru = GRU(latent_dim, return_state=True)
    _, encoder_state = encoder_gru(encoder_inputs)
    
    decoder_inputs = RepeatVector(forecast_horizon)(encoder_state)
    decoder_gru = GRU(latent_dim, return_sequences=True)
    decoder_outputs = decoder_gru(decoder_inputs, initial_state=encoder_state)
    decoder_dense = TimeDistributed(Dense(1))
    outputs = decoder_dense(decoder_outputs)
    
    model = Model(encoder_inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    model.summary()
    return model

def build_bilstm_regressor(window_size, feature_count=1):
    # Single-step prediction model (regressor) from a window of data.
    model = Sequential()
    model.add(Bidirectional(LSTM(64, return_sequences=False), input_shape=(window_size, feature_count)))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    model.summary()
    return model

def build_lstm_attention_regressor(window_size, learning_rate=0.001):
    # LSTM with an attention mechanism for regression.
    model = Sequential()
    model.add(Input(shape=(window_size, 1)))
    model.add(LSTM(64, return_sequences=True))
    model.add(AttentionLayer())
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
    model.summary()
    return model

def build_transformer_regressor(window_size, d_model=64, num_heads=4, ff_dim=64, dropout_rate=0.1, learning_rate=0.001):
    # Transformer-based regressor.
    inputs = Input(shape=(window_size, 1))
    x = Dense(d_model)(inputs)
    
    # Positional embeddings:
    positions = tf.range(start=0, limit=window_size, delta=1)
    pos_embedding = Embedding(input_dim=window_size, output_dim=d_model)(positions)
    pos_embedding = tf.expand_dims(pos_embedding, axis=0)  # shape: (1, window_size, d_model)
    x = x + pos_embedding
    
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
    attn_output = Dropout(dropout_rate)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(x + attn_output)
    
    ffn = Dense(ff_dim, activation='relu')(out1)
    ffn = Dense(d_model)(ffn)
    ffn = Dropout(dropout_rate)(ffn)
    out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn)
    
    gap = GlobalAveragePooling1D()(out2)
    outputs = Dense(1)(gap)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
    model.summary()
    return model

def iterative_forecast(model, input_sequence, forecast_horizon):
    """
    Iteratively forecasts one step at a time for regressor models.
    - model: trained regression model.
    - input_sequence: numpy array of shape (window_size, 1)
    Returns an array of forecasted values (length = forecast_horizon).
    """
    preds = []
    current_input = input_sequence.copy()
    for _ in range(forecast_horizon):
        pred = model.predict(current_input[np.newaxis, :, :], verbose=0)
        pred_value = pred[0, 0]
        preds.append(pred_value)
        current_input = np.concatenate([current_input[1:], [[pred_value]]], axis=0)
    return np.array(preds)


def get_data(ticker="TSLA", period='5y'):
    df = yf.download(ticker, period=period)[['Close']].reset_index()
    df.columns = ['Date', 'Price']
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date')
    return df

def get_metrics_split(model_name, y_train_true, y_train_pred, y_test_true, y_test_pred):
    return pd.DataFrame({
        'model': [model_name],
        'train_mae': [mean_absolute_error(y_train_true, y_train_pred)],
        'train_mse': [mean_squared_error(y_train_true, y_train_pred)],
        'train_mape': [mean_absolute_percentage_error(y_train_true, y_train_pred)],
        'test_mae': [mean_absolute_error(y_test_true, y_test_pred)],
        'test_mse': [mean_squared_error(y_test_true, y_test_pred)],
        'test_mape': [mean_absolute_percentage_error(y_test_true, y_test_pred)],
    })

def split_and_forecast(data, window_size=60, forecast_horizon=30):
    # Scale the entire series
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data.reshape(-1, 1))

    # Prepare regression sequences
    def create_regression_data(series, window_size):
        X, y = [], []
        for i in range(len(series) - window_size):
            X.append(series[i : i + window_size])
            y.append(series[i + window_size])
        return np.array(X), np.array(y)

    X, y = create_regression_data(scaled_data, window_size)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # 75% train, 25% test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)

    # Forecast from the latest available window (for next 30 days)
    latest_input = scaled_data[-window_size:].reshape(1, window_size, 1)

    return X_train, X_test, y_train, y_test, latest_input, scaler


def create_seq2seq_data(series , window_size, forecast_horizon):
        X, y = [], []
        for i in range(len(series) - window_size - forecast_horizon + 1):
            X.append(series[i:i+window_size])
            y.append(series[i+window_size:i+window_size+forecast_horizon])
        return np.array(X), np.array(y)

def create_regression_data(series , window_size):
        X, y = [], []
        for i in range(len(series) - window_size):
            X.append(series[i:i+window_size])
            y.append(series[i+window_size])
        return np.array(X), np.array(y)    



def check_if_ticker_models_exist(ticker):
    models_dir = os.path.join("models", ticker)
    return os.path.exists(models_dir) and any(f.endswith(".keras") for f in os.listdir(models_dir))




def run_full_forecasting_pipeline(ticker: str, period: str = '5y', window_size: int = 60, forecast_horizon: int = 30, epochs: int = 50, batch_size: int = 32):
    
    progress = st.progress(0)
    status = st.empty()

    status.text("Loading data...")
    progress.progress(0.1)
    rs = 42

    df = get_data(ticker, period)
    prices = df['Price'].to_numpy()

    # Scale entire series
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices.reshape(-1, 1))

    # Create regression and seq2seq data
    

    # Prepare data
    X_seq, y_seq = create_seq2seq_data(scaled , window_size, forecast_horizon)
    X_seq = X_seq.reshape(X_seq.shape[0], window_size, 1)
    y_seq = y_seq.reshape(y_seq.shape[0], forecast_horizon, 1)

    X_reg, y_reg = create_regression_data(scaled , window_size)
    X_reg = X_reg.reshape(X_reg.shape[0], window_size, 1)

    X_seq_train, X_seq_test, y_seq_train, y_seq_test = train_test_split(X_seq, y_seq, test_size=0.25, shuffle=False , random_state=rs)
    X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.25, shuffle=False , random_state=rs)

    final_input = scaled[-window_size:].reshape(1, window_size, 1)

    predictions = {}
    metrics = []
    models_dir = os.path.join("models", ticker)
    status.text("Checking if ticker models exist...")
    progress.progress(0.2)
    model_exists = check_if_ticker_models_exist(ticker)
    
    if model_exists:
        status.text("Loading pre-trained models...")
        progress.progress(0.3)

        # Load the models from the directory
        model_files = [f for f in os.listdir(models_dir) if f.endswith(".keras")]
        for model_file in model_files :
            
            model_name = model_file.split('.')[0]
            status.text(f"Loading model: {model_name}")
            model_path = os.path.join(models_dir, model_file)
            model = tf.keras.models.load_model(model_path, custom_objects={"AttentionLayer": AttentionLayer})

            if "Bidirectional" in model_name:
                decoder_input_train = np.zeros_like(y_seq_train)
                decoder_input_test = np.zeros_like(y_seq_test)
                decoder_input_future = np.zeros((1, forecast_horizon, 1))

                train_pred = model.predict([X_seq_train, decoder_input_train])
                test_pred = model.predict([X_seq_test, decoder_input_test])
                future_pred = model.predict([final_input, decoder_input_future])

                y_train_true = scaler.inverse_transform(y_seq_train.reshape(-1, 1)).flatten()
                y_train_pred = scaler.inverse_transform(train_pred.reshape(-1, 1)).flatten()
                y_test_true = scaler.inverse_transform(y_seq_test.reshape(-1, 1)).flatten()
                y_test_pred = scaler.inverse_transform(test_pred.reshape(-1, 1)).flatten()

            elif "Seq2Seq" in model_name:
                train_pred = model.predict(X_seq_train)
                test_pred = model.predict(X_seq_test)
                future_pred = model.predict(final_input)

                y_train_true = scaler.inverse_transform(y_seq_train.reshape(-1, 1)).flatten()
                y_train_pred = scaler.inverse_transform(train_pred.reshape(-1, 1)).flatten()
                y_test_true = scaler.inverse_transform(y_seq_test.reshape(-1, 1)).flatten()
                y_test_pred = scaler.inverse_transform(test_pred.reshape(-1, 1)).flatten()

            else:
                train_pred = model.predict(X_reg_train).flatten()
                test_pred = model.predict(X_reg_test).flatten()
                future_pred = iterative_forecast(model, final_input[0], forecast_horizon)

                y_train_true = scaler.inverse_transform(y_reg_train.reshape(-1, 1)).flatten()
                y_train_pred = scaler.inverse_transform(train_pred.reshape(-1, 1)).flatten()
                y_test_true = scaler.inverse_transform(y_reg_test.reshape(-1, 1)).flatten()
                y_test_pred = scaler.inverse_transform(test_pred.reshape(-1, 1)).flatten()
            status.text(f"Predicting with model: {model_name}")

            predictions[model_name] = scaler.inverse_transform(future_pred.reshape(-1, 1)).flatten()
            metrics.append(get_metrics_split(model_name, y_train_true, y_train_pred, y_test_true, y_test_pred))
        status.text("All models loaded successfully.")
        progress.progress(0.8)


    elif not model_exists:

        status.text("Training models...")
        progress.progress(0.4)
        models_dir = "./models" +"/" + ticker
        os.makedirs(models_dir, exist_ok=True)
        
        # --------- Model 1: LSTM Bidirectional Seq2Seq ---------
        model = build_lstm_bidirectional_seq2seq_model(window_size, forecast_horizon)
        model.fit([X_seq_train, np.zeros_like(y_seq_train)], y_seq_train, validation_split=0.2, epochs=epochs, batch_size=batch_size, verbose=0)
        train_pred = model.predict([X_seq_train, np.zeros_like(y_seq_train)])
        test_pred = model.predict([X_seq_test, np.zeros_like(y_seq_test)])
        future_pred = model.predict([final_input, np.zeros((1, forecast_horizon, 1))])
        name = "LSTM_Bidirectional_Seq2Seq"
        predictions[name] = scaler.inverse_transform(future_pred.reshape(-1, 1)).flatten()
        metrics.append(get_metrics_split(name, scaler.inverse_transform(y_seq_train.reshape(-1, 1)).flatten(), scaler.inverse_transform(train_pred.reshape(-1, 1)).flatten(), scaler.inverse_transform(y_seq_test.reshape(-1, 1)).flatten(), scaler.inverse_transform(test_pred.reshape(-1, 1)).flatten()))
        model.save(os.path.join(models_dir, name + ".keras"))
        status.text(f"Model {name} trained and saved.")
        progress.progress(0.45)
        # --------- Model 2: LSTM Seq2Seq ---------
        model = build_lstm_seq2seq_model(window_size, forecast_horizon)
        model.fit(X_seq_train, y_seq_train, validation_split=0.2, epochs=epochs, batch_size=batch_size, verbose=0)
        train_pred = model.predict(X_seq_train)
        test_pred = model.predict(X_seq_test)
        future_pred = model.predict(final_input)
        name = "LSTM_Seq2Seq"
        predictions[name] = scaler.inverse_transform(future_pred.reshape(-1, 1)).flatten()
        metrics.append(get_metrics_split(name, scaler.inverse_transform(y_seq_train.reshape(-1, 1)).flatten(), scaler.inverse_transform(train_pred.reshape(-1, 1)).flatten(), scaler.inverse_transform(y_seq_test.reshape(-1, 1)).flatten(), scaler.inverse_transform(test_pred.reshape(-1, 1)).flatten()))
        model.save(os.path.join(models_dir, name + ".keras"))
        status.text(f"Model {name} trained and saved.")
        progress.progress(0.5)
        # --------- Model 3: GRU Seq2Seq ---------
        model = build_gru_seq2seq_model(window_size, forecast_horizon)
        model.fit(X_seq_train, y_seq_train, validation_split=0.2, epochs=epochs, batch_size=batch_size, verbose=0)
        train_pred = model.predict(X_seq_train)
        test_pred = model.predict(X_seq_test)
        future_pred = model.predict(final_input)
        name = "GRU_Seq2Seq"
        predictions[name] = scaler.inverse_transform(future_pred.reshape(-1, 1)).flatten()
        metrics.append(get_metrics_split(name, scaler.inverse_transform(y_seq_train.reshape(-1, 1)).flatten(), scaler.inverse_transform(train_pred.reshape(-1, 1)).flatten(), scaler.inverse_transform(y_seq_test.reshape(-1, 1)).flatten(), scaler.inverse_transform(test_pred.reshape(-1, 1)).flatten()))
        model.save(os.path.join(models_dir, name + ".keras"))
        status.text(f"Model {name} trained and saved.")
        progress.progress(0.55)
        # --------- Model 4: BiLSTM Regressor ---------
        model = build_bilstm_regressor(window_size)
        model.fit(X_reg_train, y_reg_train, validation_split=0.2, epochs=epochs, batch_size=batch_size, verbose=0)
        train_pred = model.predict(X_reg_train).flatten()
        test_pred = model.predict(X_reg_test).flatten()
        future_pred = iterative_forecast(model, final_input[0], forecast_horizon)
        name = "BiLSTM_Regressor"
        predictions[name] = scaler.inverse_transform(future_pred.reshape(-1, 1)).flatten()
        metrics.append(get_metrics_split(name, scaler.inverse_transform(y_reg_train.reshape(-1, 1)).flatten(), scaler.inverse_transform(train_pred.reshape(-1, 1)).flatten(), scaler.inverse_transform(y_reg_test.reshape(-1, 1)).flatten(), scaler.inverse_transform(test_pred.reshape(-1, 1)).flatten()))
        model.save(os.path.join(models_dir, name + ".keras"))
        status.text(f"Model {name} trained and saved.")
        progress.progress(0.6)
        # --------- Model 5: LSTM + Attention ---------
        model = build_lstm_attention_regressor(window_size)
        model.fit(X_reg_train, y_reg_train, validation_split=0.2, epochs=epochs, batch_size=batch_size, verbose=0)
        train_pred = model.predict(X_reg_train).flatten()
        test_pred = model.predict(X_reg_test).flatten()
        future_pred = iterative_forecast(model, final_input[0], forecast_horizon)
        name = "LSTM_Attention_Regressor"
        predictions[name] = scaler.inverse_transform(future_pred.reshape(-1, 1)).flatten()
        metrics.append(get_metrics_split(name, scaler.inverse_transform(y_reg_train.reshape(-1, 1)).flatten(), scaler.inverse_transform(train_pred.reshape(-1, 1)).flatten(), scaler.inverse_transform(y_reg_test.reshape(-1, 1)).flatten(), scaler.inverse_transform(test_pred.reshape(-1, 1)).flatten()))
        model.save(os.path.join(models_dir, name + ".keras"))
        status.text(f"Model {name} trained and saved.")
        progress.progress(0.65)
        # --------- Model 6: Transformer Regressor ---------
        model = build_transformer_regressor(window_size)
        model.fit(X_reg_train, y_reg_train, validation_split=0.2, epochs=epochs, batch_size=batch_size, verbose=0)
        train_pred = model.predict(X_reg_train).flatten()
        test_pred = model.predict(X_reg_test).flatten()
        future_pred = iterative_forecast(model, final_input[0], forecast_horizon)
        name = "Transformer_Regressor"
        predictions[name] = scaler.inverse_transform(future_pred.reshape(-1, 1)).flatten()
        metrics.append(get_metrics_split(name, scaler.inverse_transform(y_reg_train.reshape(-1, 1)).flatten(), scaler.inverse_transform(train_pred.reshape(-1, 1)).flatten(), scaler.inverse_transform(y_reg_test.reshape(-1, 1)).flatten(), scaler.inverse_transform(test_pred.reshape(-1, 1)).flatten()))
        model.save(os.path.join(models_dir, name + ".keras"))
        status.text(f"Model {name} trained and saved.")
        progress.progress(0.7)

    # Format metrics + predictions
    status.text("Calculating metrics...")
    progress.progress(0.9)
    metrics_df = pd.concat(metrics, ignore_index=True)
    last_date = df['Date'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon)

    forecast_df = pd.DataFrame([
        {"Date": date, "Forecast": value, "Model": model_name}
        for model_name, forecast in predictions.items()
        for date, value in zip(future_dates, forecast)
    ])
    progress.progress(1.0)

    return forecast_df, metrics_df
