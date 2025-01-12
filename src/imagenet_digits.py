import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Input
from tensorflow.keras.models import Model
import tensorflow as tf

# Load digits dataset
digits = load_digits()
X = digits.images  # Shape: (n_samples, 8, 8)
y = digits.target  # Shape: (n_samples,)

# Preprocess features
X = X / 16.0  # Normalize pixel values to [0, 1]
X_resized = np.array(
    [tf.image.resize(np.repeat(img[..., np.newaxis], 3, axis=-1), (64, 64)).numpy() for img in X]
)

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Train-test split
X_train_val, X_test, y_train_val, y_test = train_test_split(X_resized, y_encoded, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

# Load pre-trained VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))

# Add custom classification layers
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

# Final model
classifier_model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the convolutional base
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
classifier_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = classifier_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=5,
    batch_size=16,
    verbose=1
)

# Evaluate on test data
test_loss, test_acc = classifier_model.evaluate(X_test, y_test, verbose=1)
print(f"\nTest Accuracy: {test_acc:.4f}")
