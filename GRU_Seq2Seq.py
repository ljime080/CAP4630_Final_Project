import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.optimizers import Adam

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------

def fetch_data(ticker='TSLA', period='5y'):
    """
    Fetches historical stock data for the given ticker and period.
    Returns the closing prices as a numpy array.
    """
    data = yf.download(ticker, period=period)
    return data['Close'].values.reshape(-1, 1)

def scale_data(data):
    """
    Scales the data to the range [0,1] using MinMaxScaler.
    Returns the scaled data and the scaler (for inverse transformations).
    """
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)
    return scaled, scaler

def create_sequences(data, n_encoder, n_decoder):
    """
    Generates sequences with a sliding window approach.
    For every sample, uses 'n_encoder' time steps as input and the following 
    'n_decoder' time steps as the target output.
    """
    X, y = [], []
    for i in range(len(data) - n_encoder - n_decoder + 1):
        X.append(data[i : i + n_encoder])
        y.append(data[i + n_encoder : i + n_encoder + n_decoder])
    return np.array(X), np.array(y)

def prepare_data(scaled_data, n_encoder, n_decoder):
    """
    Splits the scaled data into training and testing sets.
    Uses all but the last 'n_decoder' time steps as training data.
    Also creates training sequences with the specified encoder and decoder lengths.
    """
    train_data = scaled_data[:-n_decoder]
    test_data = scaled_data[-n_decoder:]
    X_train, y_train = create_sequences(train_data, n_encoder, n_decoder)
    return X_train, y_train, test_data, train_data

def build_model(n_encoder, n_decoder, latent_dim=64):
    """
    Builds and compiles a GRU-based Seq2Seq model.
    """
    # Encoder
    encoder_inputs = Input(shape=(n_encoder, 1))
    encoder_gru = GRU(latent_dim, return_state=True)
    _, encoder_state = encoder_gru(encoder_inputs)
    
    # Decoder
    decoder_inputs = RepeatVector(n_decoder)(encoder_state)
    decoder_gru = GRU(latent_dim, return_sequences=True)
    decoder_outputs = decoder_gru(decoder_inputs, initial_state=encoder_state)
    decoder_dense = TimeDistributed(Dense(1))
    outputs = decoder_dense(decoder_outputs)
    
    model = Model(encoder_inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

def plot_loss(history):
    """
    Plots training and validation loss curves.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training vs. Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.show()

def plot_prediction(actual, predicted, title, xlabel="Time step", ylabel="Stock Price"):
    """
    Plots predicted values vs actual values.
    """
    plt.figure(figsize=(8, 4))
    plt.plot(actual, label='Actual')
    plt.plot(predicted, label='Predicted', linestyle='--')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

# ------------------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------------------
def main():
    # Parameters
    ticker = 'TSLA'
    period = '5y'
    n_encoder = 30   # Number of time steps for encoder input
    n_decoder = 30   # Number of time steps for decoder output
    latent_dim = 64  # Number of GRU units
    epochs = 50
    batch_size = 16
    
    # Step 1: Fetch and scale data
    raw_data = fetch_data(ticker, period)
    scaled_data, scaler = scale_data(raw_data)
    
    # Step 2: Prepare the data (splitting into training and testing)
    X_train, y_train, test_data, train_data = prepare_data(scaled_data, n_encoder, n_decoder)
    print(f"Training samples: {X_train.shape[0]}, Input shape: {X_train.shape[1:]}, Output shape: {y_train.shape[1:]}")
    
    # Step 3: Build and summarize the GRU Seq2Seq model
    model = build_model(n_encoder, n_decoder, latent_dim)
    model.summary()
    
    # Train the model with a validation split
    history = model.fit(X_train, y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        validation_split=0.1,
                        verbose=1)
    
    # Step 4: Plot training and validation loss
    plot_loss(history)
    
    # Step 5: Plot predicted vs. actual for a training sample (e.g., first sample)
    sample_index = 0
    sample_input = X_train[sample_index:sample_index+1]  # shape: (1, n_encoder, 1)
    sample_actual = y_train[sample_index]                 # shape: (n_decoder, 1)
    sample_pred = model.predict(sample_input)
    
    # Inverse transform back to original scale
    sample_pred_inv = scaler.inverse_transform(sample_pred.reshape(-1, 1))
    sample_actual_inv = scaler.inverse_transform(sample_actual.reshape(-1, 1))
    plot_prediction(sample_actual_inv, sample_pred_inv, 'Training Sample: Predicted vs Actual')
    
    # Step 6: Test the model on the most recent 30 days
    # We use the last n_encoder days of the training data to predict the test sequence.
    test_input = train_data[-n_encoder:].reshape(1, n_encoder, 1)
    test_pred = model.predict(test_input)
    test_pred_inv = scaler.inverse_transform(test_pred.reshape(-1, 1))
    test_actual_inv = scaler.inverse_transform(test_data.reshape(-1, 1))
    plot_prediction(test_actual_inv, test_pred_inv, 'Test Set (Most Recent 30 Days): Actual vs Predicted')

if __name__ == "__main__":
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    main()
