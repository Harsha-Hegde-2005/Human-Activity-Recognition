# src/train.py
import matplotlib
matplotlib.use('Agg') # Use a non-interactive backend
import matplotlib.pyplot as plt

from data_loader import load_har_dataset
from model import build_lstm_model

def plot_comparison(histories, optimizers):
    """
    Plots a comparison of training AND validation accuracy for different optimizers.
    """
    plt.figure(figsize=(12, 8))
    for optimizer in optimizers:
        history = histories[optimizer]
        
        # Plot Validation Accuracy (Solid Line)
        plt.plot(history.history['val_accuracy'], linestyle='-', label=f'{optimizer.upper()} Val Accuracy')
        # Plot Training Accuracy (Dashed Line)
        plt.plot(history.history['accuracy'], linestyle='--', label=f'{optimizer.upper()} Train Accuracy')
    
    plt.title('Model Training vs. Validation Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('optimizer_comparison.png')
    print("\nComparison plot saved as optimizer_comparison.png")

def main():
    """
    Main function to run the model training and comparison process.
    """
    X_train, y_train, X_test, y_test = load_har_dataset()
    
    input_shape = (X_train.shape[1], X_train.shape[2])
    num_classes = y_train.shape[1]
    
    optimizers_to_compare = ['adam', 'sgd']
    training_histories = {}

    for optimizer in optimizers_to_compare:
        print(f"\n--- Training with Optimizer: {optimizer.upper()} ---")
        
        model = build_lstm_model(input_shape=input_shape, num_classes=num_classes)
        
        model.compile(optimizer=optimizer, 
                      loss='categorical_crossentropy', 
                      metrics=['accuracy'])
        
        history = model.fit(
            X_train, y_train,
            epochs=5,
            batch_size=64,
            validation_data=(X_test, y_test),
            verbose=1
        )
        training_histories[optimizer] = history
        
        # --- Evaluate and Print Final Metrics ---
        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
        train_accuracy = history.history['accuracy'][-1]
        train_loss = history.history['loss'][-1]

        print(f"\n--- Final Results for {optimizer.upper()} ---")
        print(f"Final Training Accuracy: {train_accuracy*100:.2f}%")
        print(f"Final Training Error (Loss): {train_loss:.4f}")
        print(f"Final Test Accuracy: {test_accuracy*100:.2f}%")
        print(f"Final Test Error (Loss): {test_loss:.4f}")

    plot_comparison(training_histories, optimizers_to_compare)

if __name__ == "__main__":
    main()