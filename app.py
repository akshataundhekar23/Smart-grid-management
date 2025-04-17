import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('C:\Users\Lenovo\OneDrive\New folder\smart grid\smart_grid_stability_augmented.csv')

# Helper function for scaling and splitting
def prepare_data(features, target, test_size=0.2):
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(data[features])

    if isinstance(target, str):  # Classification
        y = data[target].map({'unstable': 0, 'stable': 1}).values
        y = to_categorical(y)
    else:  # Regression
        scaler_y = MinMaxScaler()
        y_scaled = scaler_y.fit_transform(data[target].values.reshape(-1, 1))
        return train_test_split(X_scaled, y_scaled, test_size=test_size, random_state=42), scaler_X, scaler_y

    return train_test_split(X_scaled, y, test_size=test_size, random_state=42), scaler_X

# Helper function for building ANN models
def build_ann(input_dim, output_dim, output_activation, loss, optimizer='adam'):
    model = Sequential([
        Dense(64, input_dim=input_dim, activation='relu'),
        Dense(32, activation='relu'),
        Dense(output_dim, activation=output_activation)
    ])
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy' if output_activation == 'softmax' else 'mae'])
    return model

# Function to plot training history
def plot_training_history(history, task_type):
    plt.figure(figsize=(12, 5))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{task_type} - Loss')
    plt.legend()

    # Plot accuracy/MAE
    plt.subplot(1, 2, 2)
    if 'accuracy' in history.history:
        plt.plot(history.history['accuracy'], label='Train Accuracy')
        plt.plot(history.history['val_accuracy'], label='Val Accuracy')
        metric = 'Accuracy'
    else:
        plt.plot(history.history['mae'], label='Train MAE')
        plt.plot(history.history['val_mae'], label='Val MAE')
        metric = 'MAE'

    plt.title(f'{task_type} - {metric}')
    plt.legend()
    plt.show()