import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD, RMSprop


# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess data
X_train = X_train[..., np.newaxis] / 255.0  # Normalize and add channel dimension
X_test = X_test[..., np.newaxis] / 255.0
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Define parameters to test
params = [
    {'optimizer': Adam(learning_rate=0.001), 'filters': 32, 'kernel_size': (3, 3), 'pooling_size': (2, 2)},
    # {'optimizer': Adam(learning_rate=0.001), 'filters': 64, 'kernel_size': (3, 3), 'pooling_size': (2, 2)},
    {'optimizer': Adam(learning_rate=0.001), 'filters': 64, 'kernel_size': (5, 5), 'pooling_size': (2, 2)},
]

results = []

# Train and evaluate models with different parameters
for param in params:
    print(f"Training with params: {param}")
    model = Sequential([
        Conv2D(param['filters'], param['kernel_size'], activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(param['pooling_size']),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),  # Add Dropout for regularization
        Dense(10, activation='softmax')
    ])

    model.compile(optimizer=param['optimizer'], loss='categorical_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=1
    )

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    results.append({'params': param.values(), 'val_loss': val_loss, 'val_acc': val_acc})
    print(f"Validation Accuracy: {val_acc:.4f}")

# Print results
print("\nSummary of Results:")
for i, result in enumerate(results):
    print(f"Params {i+1}: {result['params']} -> Validation Accuracy: {result['val_acc']:.4f}")

# Plot results
plt.figure(figsize=(10, 6))
for i, result in enumerate(results):
    plt.bar(i, result['val_acc'], label=f"Params {i+1}")
plt.xticks(range(len(results)), [f"Params {i+1}" for i in range(len(results))], rotation=45)
plt.title("Validation Accuracy for Different Parameters")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
