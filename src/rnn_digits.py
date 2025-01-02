import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.datasets import load_digits
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Input
from tensorflow.keras.callbacks import EarlyStopping

# Load and preprocess the data
digits = load_digits()
X = digits.data
y = digits.target

print('X:\n', X)
print('y:\n', y)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape for RNN
X_scaled_rnn = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
print('X_scaled:\n', X_scaled_rnn)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train-test split
X_train_val, X_test, y_train_val, y_test = train_test_split(X_scaled_rnn, y_encoded, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Define RNN Model
rnn_model = Sequential([
    SimpleRNN(64, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])),
    Dense(10, activation='softmax') # 10 classes for digits 0-9
])

rnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

rnn_history = rnn_model.fit(
    X_train, y_train,
    epochs=50, batch_size=16,
    validation_data=(X_val, y_val),
    verbose=1,
    callbacks=[early_stopping]
)

# RNN Evaluation
y_pred_rnn = np.argmax(rnn_model.predict(X_test), axis=1)
rnn_acc = accuracy_score(y_test, y_pred_rnn)

print("\nRNN Classification Report:")
target_names = [str(i) for i in range(len(np.unique(y)))]
print(classification_report(y_test, y_pred_rnn, target_names=target_names))

# Train-test split for Dense model (reshape back for dense network)
X_train_dense, X_test_dense, y_train_dense, y_test_dense = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
X_train_dense, X_val_dense, y_train_dense, y_val_dense = train_test_split(X_train_dense, y_train_dense, test_size=0.2, random_state=42)

# Define Dense Model
dense_model = Sequential([
    Input(shape=(X_train_dense.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')
])

dense_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

dense_history = dense_model.fit(
    X_train_dense, y_train_dense,
    epochs=50, batch_size=16,
    validation_data=(X_val_dense, y_val_dense),
    verbose=1,
    callbacks=[early_stopping]
)

# Dense Model Evaluation
y_pred_dense = np.argmax(dense_model.predict(X_test_dense), axis=1)
dense_acc = accuracy_score(y_test_dense, y_pred_dense)

print("\nDense Classification Report:")
print(classification_report(y_test_dense, y_pred_dense, target_names=target_names))

# Print Summary
print("\nComparison Summary:")
print(f"RNN Model Test Accuracy: {rnn_acc:.4f}")
print(f"Dense Model Test Accuracy: {dense_acc:.4f}")

# Plot Training History Comparison
plt.figure(figsize=(14, 6))

# Loss comparison
plt.subplot(1, 2, 1)
plt.plot(rnn_history.history['loss'], label='RNN Training Loss')
plt.plot(rnn_history.history['val_loss'], label='RNN Validation Loss')
plt.plot(dense_history.history['loss'], label='Dense Training Loss', linestyle='--')
plt.plot(dense_history.history['val_loss'], label='Dense Validation Loss', linestyle='--')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Comparison')
plt.legend()

# Accuracy comparison
plt.subplot(1, 2, 2)
plt.plot(rnn_history.history['accuracy'], label='RNN Training Accuracy')
plt.plot(rnn_history.history['val_accuracy'], label='RNN Validation Accuracy')
plt.plot(dense_history.history['accuracy'], label='Dense Training Accuracy', linestyle='--')
plt.plot(dense_history.history['val_accuracy'], label='Dense Validation Accuracy', linestyle='--')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy Comparison')
plt.legend()

plt.tight_layout()
plt.show()
