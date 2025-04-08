import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf


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

def build_transformer_model(window_size, d_model=64, num_heads=4, ff_dim=64, dropout_rate=0.1, learning_rate=0.001):
    inputs = tf.keras.Input(shape=(window_size, 1))
    # Project input to d_model dimensions
    x = Dense(d_model)(inputs)
    
    # Create and add positional embeddings
    positions = tf.range(start=0, limit=window_size, delta=1)
    pos_embedding = tf.keras.layers.Embedding(input_dim=window_size, output_dim=d_model)(positions)
    pos_embedding = tf.expand_dims(pos_embedding, axis=0)  # shape: (1, window_size, d_model)
    x = x + pos_embedding

    # Transformer encoder block
    attn_output = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
    attn_output = tf.keras.layers.Dropout(dropout_rate)(attn_output)
    out1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
    
    ffn = Dense(ff_dim, activation='relu')(out1)
    ffn = Dense(d_model)(ffn)
    ffn = tf.keras.layers.Dropout(dropout_rate)(ffn)
    out2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(out1 + ffn)
    
    # Global average pooling and final dense layer for output
    gap = tf.keras.layers.GlobalAveragePooling1D()(out2)
    outputs = Dense(1)(gap)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mean_squared_error")
    model.summary()
    return model

def create_and_fit_model(model, X_train, y_train, epochs=50, batch_size=16, validation_split=0.1):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
    return history

def plot_loss_history(history, filename="plots/transformer_training_loss.png"):
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title("Transformer Training Loss History")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(filename)
    plt.show()

def plot_train_results(model, X_train, y_train, scaler, filename="plots/transformer_training_results.png"):
    predictions = model.predict(X_train)
    predicted_prices = scaler.inverse_transform(predictions)
    actual_prices = scaler.inverse_transform(y_train)
    plt.figure(figsize=(8, 4))
    plt.plot(actual_prices, label="Actual Prices")
    plt.plot(predicted_prices, label="Predicted Prices")
    plt.title("Transformer Training Data Forecast")
    plt.xlabel("Time (days)")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.savefig(filename)
    plt.show()

def plot_test_results(model, X_test, y_test, scaler, filename="plots/transformer_test_results.png"):
    predictions = model.predict(X_test)
    predicted_prices = scaler.inverse_transform(predictions)
    actual_prices = scaler.inverse_transform(y_test)
    plt.figure(figsize=(8, 4))
    plt.plot(actual_prices, label="Actual Prices")
    plt.plot(predicted_prices, label="Predicted Prices")
    plt.title("Transformer Test Data Forecast")
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

    model = build_transformer_model(window_size)
    history = create_and_fit_model(model, X_train, y_train, epochs=epochs, batch_size=batch_size)
    
    plot_loss_history(history, filename="plots/transformer_training_loss.png")
    plot_train_results(model, X_train, y_train, scaler, filename="plots/transformer_training_results.png")
    plot_test_results(model, X_test, y_test, scaler, filename="plots/transformer_test_results.png")
    
    model.save("models/final_model_transformer.h5")
    print("Transformer model saved as models/final_model_transformer.h5")

if __name__ == "__main__":
    main()
