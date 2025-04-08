
# Import all necessary packages.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf


from tensorflow.keras.layers import Input, LSTM, GRU, Dense, RepeatVector, TimeDistributed, Layer, Bidirectional, MultiHeadAttention, Dropout, LayerNormalization, Embedding, GlobalAveragePooling1D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

import os
import random


os.environ['PYTHONHASHSEED'] = '42'
os.environ['TF_DETERMINISTIC_OPS'] = '1' 

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)




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



def create_seq2seq_data(series, window_size, forecast_horizon):
    """
    Creates sliding window sequences for seq2seq models.
    - series: numpy array of shape (n_samples, 1)
    """
    X, y = [], []
    for i in range(len(series) - window_size - forecast_horizon + 1):
        X.append(series[i : i + window_size])
        y.append(series[i + window_size : i + window_size + forecast_horizon])
    return np.array(X), np.array(y)

def create_regression_data(series, window_size):
    """
    Creates sliding windows for one-step regression models.
    """
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i : i + window_size])
        y.append(series[i + window_size])
    return np.array(X), np.array(y)

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




def get_data(ticker="TSLA" , period = '5y'):
    # 1. Load Tesla stock data for the past 5 years.
    ticker = "TSLA"
    data = yf.download(ticker, period="5y")
    data = data[['Close']].dropna()

    return data




def get_metrics(model_name , y_true , y_pred):
    mae = mean_absolute_error(y_true , y_pred)
    mse = mean_squared_error(y_true , y_pred)
    mape = mean_absolute_percentage_error(y_true , y_pred)

    ret = pd.DataFrame({
        'model':[model_name],
        'mse':[mse],
        'mae':[mae],
        'mape':[mape]
    })

    return ret



ticker="TSLA" 
period = '5y'

data = get_data(ticker , period)

# 2. Define forecasting parameters and split the data.
forecast_horizon = 30     # Last 30 days for testing.
window_size = 60          # Use the past 60 days as input.
train_data = data.iloc[:-forecast_horizon]
test_data = data.iloc[-forecast_horizon:]

# 3. Scale the data.
print("Scaling data")
scaler = MinMaxScaler(feature_range=(0,1))
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)

print("Creating seq2seq data")
# 4. Prepare data for sequence-to-sequence models (Group A).
X_seq, y_seq = create_seq2seq_data(train_scaled, window_size, forecast_horizon)
# For testing, use the last window from training as input.
X_seq_test = train_scaled[-window_size:].reshape(1, window_size, 1)
y_seq_test = test_scaled.reshape(1, forecast_horizon, 1)

print("Preparing data for regression models")
# 5. Prepare data for regression models (Group B).
X_reg, y_reg = create_regression_data(train_scaled, window_size)
# For testing the regressor, use the last window from training.
X_reg_test = train_scaled[-window_size:].reshape(window_size, 1)

# 6. Set training parameters.
EPOCHS = 50
BATCH_SIZE = 32

# To store training history and predictions.
history_dict = {}
predictions = {}



# Model 1: LSTM Bidirectional Seq2Seq
print("Building LSTM Bidirectional Seq2Seq model")
model_lstm_bidir = build_lstm_bidirectional_seq2seq_model(window_size, forecast_horizon, feature_count=1)
# For this model, create a decoder input (using zeros during training).
decoder_input_train = np.zeros_like(y_seq)  # shape: (n_samples, forecast_horizon, 1)

print("Fitting LSTM Bidirectional Seq2Seq model")
history_lstm_bidir = model_lstm_bidir.fit([X_seq, decoder_input_train], y_seq,
                                            validation_split=0.2,
                                            epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
history_dict['LSTM_Bidirectional_Seq2Seq'] = history_lstm_bidir.history

print("Predicting forecasts (LSTM Bidirectional Seq2Seq)")
# For prediction, still need to supply decoder input
decoder_input_test = np.zeros((1, forecast_horizon, 1))
pred_lstm_bidir = model_lstm_bidir.predict([X_seq_test, decoder_input_test])
predictions['LSTM_Bidirectional_Seq2Seq'] = scaler.inverse_transform(pred_lstm_bidir.reshape(-1, 1)).flatten()

# Model 2: LSTM Seq2Seq
print("Building LSTM Seq2Seq model")
model_lstm_seq2seq = build_lstm_seq2seq_model(window_size, forecast_horizon)
print("Fitting LSTM Seq2Seq model")
history_lstm_seq2seq = model_lstm_seq2seq.fit(X_seq, y_seq,
                                              validation_split=0.2,
                                              epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
history_dict['LSTM_Seq2Seq'] = history_lstm_seq2seq.history

print("Predicting forecasts (LSTM Seq2Seq)")
pred_lstm_seq2seq = model_lstm_seq2seq.predict(X_seq_test)
predictions['LSTM_Seq2Seq'] = scaler.inverse_transform(pred_lstm_seq2seq.reshape(-1, 1)).flatten()

# Model 3: GRU Seq2Seq
print("Building GRU Seq2Seq model")
model_gru_seq2seq = build_gru_seq2seq_model(window_size, forecast_horizon, latent_dim=64)
print("Fitting GRU Seq2Seq model")
history_gru_seq2seq = model_gru_seq2seq.fit(X_seq, y_seq,
                                            validation_split=0.2,
                                            epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
history_dict['GRU_Seq2Seq'] = history_gru_seq2seq.history

print("Predicting forecasts (GRU Seq2Seq)")
pred_gru_seq2seq = model_gru_seq2seq.predict(X_seq_test)
predictions['GRU_Seq2Seq'] = scaler.inverse_transform(pred_gru_seq2seq.reshape(-1, 1)).flatten()



# Model 4: BiLSTM Regressor
print("Building BiLSTM Regressor model")
model_bilstm = build_bilstm_regressor(window_size, feature_count=1)
print("Fitting BiLSTM Regressor model")
history_bilstm = model_bilstm.fit(X_reg, y_reg,
                                  validation_split=0.2,
                                  epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
history_dict['BiLSTM_Regressor'] = history_bilstm.history

print("Predicting forecasts (BiLSTM Regressor)")
pred_bilstm = iterative_forecast(model_bilstm, X_reg_test, forecast_horizon)
predictions['BiLSTM_Regressor'] = scaler.inverse_transform(pred_bilstm.reshape(-1, 1)).flatten()

# Model 5: LSTM Attention Regressor
print("Building LSTM Attention Regressor model")
model_lstm_attention = build_lstm_attention_regressor(window_size, learning_rate=0.001)
print("Fitting LSTM Attention Regressor model")
history_lstm_attention = model_lstm_attention.fit(X_reg, y_reg,
                                                  validation_split=0.2,
                                                  epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
history_dict['LSTM_Attention_Regressor'] = history_lstm_attention.history

print("Predicting forecasts (LSTM Attention Regressor)")
pred_lstm_attention = iterative_forecast(model_lstm_attention, X_reg_test, forecast_horizon)
predictions['LSTM_Attention_Regressor'] = scaler.inverse_transform(pred_lstm_attention.reshape(-1, 1)).flatten()

# Model 6: Transformer Regressor
print("Building Transformer Regressor model")
model_transformer = build_transformer_regressor(window_size, d_model=64, num_heads=4, ff_dim=64, dropout_rate=0.1, learning_rate=0.001)
print("Fitting Transformer Regressor model")
history_transformer = model_transformer.fit(X_reg, y_reg,
                                            validation_split=0.2,
                                            epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
history_dict['Transformer_Regressor'] = history_transformer.history

print("Predicting forecasts (Transformer Regressor)")
pred_transformer = iterative_forecast(model_transformer, X_reg_test, forecast_horizon)
predictions['Transformer_Regressor'] = scaler.inverse_transform(pred_transformer.reshape(-1, 1)).flatten()

# Get actual test values (inverse-scaled).
actual_test = scaler.inverse_transform(test_scaled.reshape(-1, 1)).flatten()


model_lstm_bidir.save("models/model_lstm_bidir_min_max.h5")
model_lstm_seq2seq.save("models/model_lstm_seq2seq_min_max.h5")
model_gru_seq2seq.save("models/model_gru_seq2seq_min_max.h5")
model_bilstm.save("models/model_bilstm_regressor_min_max.h5")
model_lstm_attention.save("models/model_lstm_attention_regressor_min_max.h5")
model_transformer.save("models/model_transformer_regressor_min_max.h5")


models = list(history_dict.keys())
plt.figure(figsize=(15, 10))

print("Plotting training and validation losses")
for i, model_name in enumerate(models, 1):
    plt.subplot(3, 2, i)
    train_loss = history_dict[model_name]['loss']
    val_loss = history_dict[model_name]['val_loss']
    epochs_range = range(1, len(train_loss) + 1)
    plt.plot(epochs_range, train_loss, label='Train Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.title(model_name)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
plt.tight_layout()
plt.savefig("plots/model_loss_using_min_max.png" , dpi = 300)
plt.show()



test_results = pd.DataFrame()

plt.figure(figsize=(15, 10))

# Plot the actual test data (using a distinct color like black and a thicker line).
plt.plot(actual_test, label='Actual', color='black', linewidth=2)

# Loop through each model's predictions and plot them on the same figure.
for model_name, pred in predictions.items():
    model_results = get_metrics(model_name , actual_test , pred)
    test_results = pd.concat([test_results , model_results])
    plt.plot(pred, label=model_name)

plt.xlabel("Time Step")
plt.ylabel("Stock Price")
plt.title("Model Predictions vs Actual")
plt.legend()
plt.savefig("plots/model_predictions_using_min_max.png" , dpi = 300)
plt.show()


test_results.to_csv("metrics/metrics_results_minmax.csv" , index=False)
