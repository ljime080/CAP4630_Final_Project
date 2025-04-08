import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input, Layer
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf


class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create trainable weights for the attention mechanism
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

# -------------------------------
# Helper Functions
# -------------------------------
def get_data(ticker="TSLA", period="5y"):
    data = yf.download(ticker, period=period)
    close_data = data[['Close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_data)
    return close_data, scaled_data, scaler

def shape_data(scaled_data, window_size=60, test_size=30):
    def create_dataset(data, window_size):
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:i+window_size])
            y.append(data[i+window_size])
        return np.array(X), np.array(y)
    
    train_data = scaled_data[:-test_size]
    test_data = scaled_data[-(test_size + window_size):]
    X_train, y_train = create_dataset(train_data, window_size)
    X_test, y_test = create_dataset(test_data, window_size)
    return X_train, y_train, X_test, y_test

def build_model(window_size, learning_rate=0.001):
    model = Sequential()
    model.add(Input(shape=(window_size, 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(AttentionLayer())
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mean_squared_error")
    model.summary()
    return model

def create_and_fit_model(model, X_train, y_train, epochs=50, batch_size=16, validation_split=0.1):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
    return history

def plot_loss_history(history, filename="plots/lstm_training_loss.png"):
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title("LSTM Training Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(filename)
    plt.show()

def plot_train_results(model, X_train, y_train, scaler, filename="plots/lstm_training_results.png"):
    predictions = model.predict(X_train)
    predicted_prices = scaler.inverse_transform(predictions)
    actual_prices = scaler.inverse_transform(y_train)
    plt.figure(figsize=(8, 4))
    plt.plot(actual_prices, label="Actual Prices")
    plt.plot(predicted_prices, label="Predicted Prices")
    plt.title("LSTM Training Data Forecast")
    plt.xlabel("Time (days)")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.savefig(filename)
    plt.show()

def plot_test_results(model, X_test, y_test, scaler, filename="plots/lstm_test_results.png"):
    predictions = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predictions)
    actual_prices = scaler.inverse_transform(y_test)
    plt.figure(figsize=(8, 4))
    plt.plot(actual_prices, label="Actual Prices")
    plt.plot(predicted_prices, label="Predicted Prices")
    plt.title("LSTM Test Data Forecast")
    plt.xlabel("Time (days)")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.savefig(filename)
    plt.show()


def main():

    window_size = 60
    test_size = 30
    epochs = 50
    batch_size = 16

    close_data, scaled_data, scaler = get_data(ticker="TSLA", period="5y")
    X_train, y_train, X_test, y_test = shape_data(scaled_data, window_size=window_size, test_size=test_size)
    
    model = build_model(window_size)
    history = create_and_fit_model(model, X_train, y_train, epochs=epochs, batch_size=batch_size)
    
    plot_loss_history(history, filename="plots/lstm_training_loss.png")
    plot_train_results(model, X_train, y_train, scaler, filename="plots/lstm_training_results.png")
    plot_test_results(model, X_test, y_test, scaler, filename="plots/lstm_test_results.png")
    
    model.save("models/final_model_lstm.h5")
    print("LSTM with Attention model saved as models/final_model_lstm.h5")

if __name__ == "__main__":
    main()
