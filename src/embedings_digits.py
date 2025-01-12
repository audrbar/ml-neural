import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# Load the digits dataset
digits = load_digits()
X = digits.images  # Shape: (n_samples, 8, 8)
y = digits.target  # Shape: (n_samples,)

# Normalize and preprocess
X = X / 16.0  # Normalize to [0, 1]
X = np.array([tf.image.resize(img[..., np.newaxis], (224, 224)).numpy() for img in X])  # Resize to (224, 224, 1)
X = np.repeat(X, 3, axis=-1)  # Convert grayscale to RGB

# Train-test split
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# Load pre-trained VGG16 model and extract embeddings
base_model = VGG16(weights='imagenet', include_top=True)
embedding_model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

# Extract embeddings
X_train_embeddings = embedding_model.predict(X_train)
X_val_embeddings = embedding_model.predict(X_val)
X_test_embeddings = embedding_model.predict(X_test)

# Reshape embeddings for RNN input
X_train_embeddings_rnn = X_train_embeddings.reshape(X_train_embeddings.shape[0], 1, -1)
X_val_embeddings_rnn = X_val_embeddings.reshape(X_val_embeddings.shape[0], 1, -1)
X_test_embeddings_rnn = X_test_embeddings.reshape(X_test_embeddings.shape[0], 1, -1)

# Define RNN model
rnn_model = Sequential([
    SimpleRNN(64, activation='relu', input_shape=(X_train_embeddings_rnn.shape[1], X_train_embeddings_rnn.shape[2])),
    Dense(10, activation='softmax')  # 10 classes for digits 0-9
])

rnn_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the RNN
rnn_history = rnn_model.fit(
    X_train_embeddings_rnn, y_train,
    validation_data=(X_val_embeddings_rnn, y_val),
    epochs=20, batch_size=32,
    verbose=1,
    callbacks=[early_stopping]
)

# Evaluate the RNN
y_pred_rnn = np.argmax(rnn_model.predict(X_test_embeddings_rnn), axis=1)
rnn_acc = accuracy_score(y_test, y_pred_rnn)

print("\nRNN Classification Report:")
target_names = [str(i) for i in range(10)]
print(classification_report(y_test, y_pred_rnn, target_names=target_names))

print(f"\nRNN Test Accuracy: {rnn_acc:.4f}")
