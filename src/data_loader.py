# src/data_loader.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

def load_har_dataset():
    """
    Loads the UCI-HAR dataset from the CSV files.

    Returns:
        X_train, y_train, X_test, y_test: Processed and reshaped data splits.
    """
    print("Loading data from CSV files...")
    
    # Load the training and test data
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    
    # Separate features (X) from labels (y)
    X_train = train_df.drop('Activity', axis=1).values
    y_train_labels = train_df['Activity'].values
    
    X_test = test_df.drop('Activity', axis=1).values
    y_test_labels = test_df['Activity'].values
    
    # --- Process the Labels ---
    # The labels are strings ('WALKING', etc.), so we need to convert them to numbers.
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train_labels)
    y_test_encoded = label_encoder.transform(y_test_labels)
    
    # Convert numerical labels to one-hot encoded vectors
    # e.g., '2' becomes [0, 0, 1, 0, 0, 0]
    num_classes = len(label_encoder.classes_)
    y_train = to_categorical(y_train_encoded, num_classes)
    y_test = to_categorical(y_test_encoded, num_classes)
    
    # --- Reshape Data for LSTM ---
    # The LSTM model expects input in the shape: [samples, timesteps, features].
    # Our data is currently 2D [samples, features], so we add a new dimension for timesteps.
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))
    
    print("Data loading and preprocessing complete.")
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    
    return X_train, y_train, X_test, y_test