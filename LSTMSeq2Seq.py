import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, RepeatVector, TimeDistributed, Dense
from tensorflow.keras.optimizers import Adam

# Set seeds for reproducibility
np.random.seed(42)

# Parameters
TICKER = 'TSLA'
PERIOD = '5y'
INPUT_LEN = 30
OUTPUT_LEN = 30
NUM_FORECASTS = 10
EPOCHS = 50
BATCH_SIZE = 64

def download_data(ticker, period='5y'):
    print(f"Downloading data for {ticker}...")
    df = yf.download(ticker, period=period)
    return df['Close'].values.reshape(-1, 1)

def create_sequences(data, input_len=30, output_len=30):
    X, y = [], []
    for i in range(len(data) - input_len - output_len):
        X.append(data[i:i+input_len])
        y.append(data[i+input_len:i+input_len+output_len])
    return np.array(X), np.array(y)

def build_seq2seq_model(input_len, output_len):
    encoder_inputs = Input(shape=(input_len, 1))
    encoder_lstm = LSTM(64, return_state=True)
    encoder_output, state_h, state_c = encoder_lstm(encoder_inputs)
    encoder_states = [state_h, state_c]

    decoder_inputs = RepeatVector(output_len)(encoder_output)
    decoder_lstm = LSTM(64, return_sequences=True)
    decoder_outputs = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_outputs = TimeDistributed(Dense(1))(decoder_outputs)

    model = Model(encoder_inputs, decoder_outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def plot_forecasts(model, X_test, y_test, scaler, num_forecasts=10):
    fig , axs = plt.subplots(5,2 , figsize  = (40,25)) 
    axs = axs.flatten()
    for i in range(num_forecasts):

        # Get the true future prices for the i-th sample
        true_trend = scaler.inverse_transform(y_test[i])
        axs[i].plot(range(len(true_trend)), true_trend, color='black', linewidth=2, label='true trend')

        # Get model's forecast for the same input
        forecast = model.predict(X_test[i:i+1])
        forecast = scaler.inverse_transform(forecast[0])
        axs[i].plot(range(len(forecast)), forecast.flatten(), label='forecast')

        axs[i].set_title(f"LSTM-Seq2Seq Forecast vs True Trend (Sample {i+1})")
        axs[i].set_xlabel("Time (in Days)")
        axs[i].set_ylabel("Close Price")
        axs[i].grid(True)
    
    plt.legend()
    plt.tight_layout()
    plt.savefig('forecasts.png')
    plt.show()

def main():
    # Download and normalize data
    data = download_data(TICKER, PERIOD)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # Prepare sequences
    X, y = create_sequences(scaled_data, INPUT_LEN, OUTPUT_LEN)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Build and train model
    model = build_seq2seq_model(INPUT_LEN, OUTPUT_LEN)
    model.summary()
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.1)

    # Forecast and plot
    plot_forecasts(model, X_test, y_test, scaler, num_forecasts=NUM_FORECASTS)

if __name__ == "__main__":
    main()
