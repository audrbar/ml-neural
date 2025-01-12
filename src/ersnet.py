import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Add channel dimension and normalize
X_train = X_train[..., np.newaxis] / 255.0
X_test = X_test[..., np.newaxis] / 255.0

# Convert grayscale to RGB and resize to (224, 224, 3)
X_train_resized = np.array([tf.image.resize(np.repeat(img, 3, axis=-1), (64, 64)).numpy() for img in X_train])
X_test_resized = np.array([tf.image.resize(np.repeat(img, 3, axis=-1), (64, 64)).numpy() for img in X_test])

# One-hot encode labels
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Split training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train_resized, y_train, test_size=0.2, random_state=42)

base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(
    X_train, y_train, epochs=5, validation_data=(X_val, y_val), batch_size=16, verbose=1
)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test_resized, y_test, verbose=1)
print(f"Test Accuracy: {test_acc:.4f}")
