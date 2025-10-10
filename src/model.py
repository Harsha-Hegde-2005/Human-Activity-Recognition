# src/model.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_lstm_model(input_shape, num_classes):
    """
    Builds the LSTM model architecture.
    
    Args:
        input_shape (tuple): Shape of the input data (timesteps, features).
        num_classes (int): Number of output classes.
        
    Returns:
        An uncompiled Keras model.
    """
    model = Sequential([
        # The input shape will be (1, 561) for our dataset
        LSTM(100, input_shape=input_shape),
        Dropout(0.5),
        Dense(100, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    print("Model architecture built successfully.")
    return model