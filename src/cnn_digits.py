import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.python.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

# Load data
digits = load_digits()
X = digits.images  # (8x8 pixels)
y = digits.target

# Preprocessing
X = X / 16.0  # Normalize pixel values to [0, 1]
X = X[..., np.newaxis]  # Reshape to 8x8x1 for CNN input
ohe = OneHotEncoder(sparse_output=False)
y = ohe.fit_transform(y.reshape(-1, 1))

# Train-test split
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# Define parameters to test
params = [
    # {'optimizer': Adam(learning_rate=0.001), 'filters': 32, 'kernel_size': (3, 3), 'pooling_size': (2, 2)},
    # {'optimizer': Adam(learning_rate=0.001), 'filters': 32, 'kernel_size': (4, 4), 'pooling_size': (2, 2)},
    # {'optimizer': SGD(learning_rate=0.001), 'filters': 64, 'kernel_size': (3, 3), 'pooling_size': (2, 2)},
    # {'optimizer': SGD(learning_rate=0.01), 'filters': 64, 'kernel_size': (4, 4), 'pooling_size': (2, 2)},
    {'optimizer': Adam(learning_rate=0.001), 'filters': 32, 'kernel_size': (3, 3), 'pooling_size': (2, 2)},
    {'optimizer': Adam(learning_rate=0.01), 'filters': 32, 'kernel_size': (5, 5), 'pooling_size': (2, 2)},
    {'optimizer': Adam(learning_rate=0.001), 'filters': 64, 'kernel_size': (3, 3), 'pooling_size': (2, 2)},
    {'optimizer': Adam(learning_rate=0.01), 'filters': 64, 'kernel_size': (3, 3), 'pooling_size': (2, 2)},
]

results = []

# Train and evaluate models with different parameters
for param in params:
    print(f"Training with params: {param}")
    # model = Sequential([
    #     Conv2D(param['filters'], param['kernel_size'], activation='relu', input_shape=(8, 8, 1)),
    #     Conv2D(param['filters'], param['kernel_size'], activation='relu'),  # Additional Conv2D layer
    #     MaxPooling2D(param['pooling_size']),
    #     Flatten(),
    #     Dense(128, activation='relu'),  # Increased Dense layer size
    #     Dropout(0.5),  # Add Dropout for regularization
    #     Dense(10, activation='softmax')
    # ])
    model = Sequential([
        Conv2D(param['filters'], param['kernel_size'], activation='relu', input_shape=(8, 8, 1)),
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
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
        verbose=0  # Suppress training logs
    )

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    results.append({'params': param.values(), 'val_loss': val_loss, 'val_acc': val_acc})
    print(f"Validation Accuracy: {val_acc:.4f}")

# Print results
print("\nSummary of Results:")
for result in results:
    print(f"Params: {result['params']} -> Validation Accuracy: {result['val_acc']:.4f}")

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
