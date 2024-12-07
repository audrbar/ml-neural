import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf

# Load the data
data = pd.read_csv("/Users/audrius/Documents/VCSPython/ml-neural-net/data/diabetes.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Split the data
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# Scaling the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# EarlyStopping setup
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# SGD with momentum optimizer
sgd_optimizer = SGD(learning_rate=0.01, momentum=0.9)

# PolynomialDecay learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=0.1,
    decay_steps=1000,
    end_learning_rate=0.01,
    power=1.0
)


# Model training function
def build_and_train_model(optimizer, optimizer_name, epochs):
    print(f"Training model with {optimizer_name} optimizer")
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=16, verbose=1, callbacks=[early_stopping])
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    # Predictions
    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > 0.4).astype(int)

    # Calculate accuracy score and confusion matrix
    acc_score = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    return history, test_loss, test_acc, optimizer_name, acc_score, cm


# List of optimizers with epoch numbers
optimizers = [
    (sgd_optimizer, "SGD", 100),
    (Adam(), "Adam", 100),
    (RMSprop(), "RMSprop", 100)
]

# Store results
history_dict = {}
test_accuracies = []

for optimizer, opt_name, epochs in optimizers:
    history, test_loss, test_acc, optimizer_name, acc_score, cm = build_and_train_model(optimizer, opt_name, epochs)
    history_dict[opt_name] = {
        'history': history,
        'test_loss': test_loss,
        'test_acc': test_acc,
        'acc_score': acc_score,
        'confusion_matrix': cm
    }
    test_accuracies.append((opt_name, test_loss, test_acc))

    # Print the results
    print(f"\nResults for {opt_name} optimizer:")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Accuracy Score: {acc_score:.4f}")
    print(f"Confusion Matrix:\n{cm}")


# Plotting function
def plot_metrics(history_dict, test_accuracies):
    plt.figure(figsize=(14, 8))

    # Loss plot
    plt.subplot(2, 1, 1)
    for opt_name, values in history_dict.items():
        history = values['history']
        test_loss = values['test_loss']
        plt.plot(history.history['loss'], label=f'{opt_name} Train Loss ({test_loss:.4f})')
        plt.plot(history.history['val_loss'], label=f'{opt_name} Val Loss ({test_loss:.4f})')
        plt.legend(title="Loss", fontsize=10, title_fontsize=12)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss during Training')

    # Accuracy plot
    plt.subplot(2, 1, 2)
    for opt_name, values in history_dict.items():
        history = values['history']
        test_acc = values['test_acc']
        plt.plot(history.history['accuracy'], label=f'{opt_name} Train Accuracy ({test_acc:.4f})')
        plt.plot(history.history['val_accuracy'], label=f'{opt_name} Val Accuracy ({test_acc:.4f})')
        plt.legend(title="Accuracy", fontsize=10, title_fontsize=12)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy during Training')

    # Show the plots
    plt.tight_layout()
    plt.show()


# Plot the graphs
plot_metrics(history_dict, test_accuracies)
