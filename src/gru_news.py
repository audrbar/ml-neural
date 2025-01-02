import re
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.src.utils import pad_sequences, to_categorical
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Input, Embedding, GRU, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


# Custom preprocessor to exclude tokens that are only digits
def preprocessor(text):
    # Remove tokens that are entirely numeric
    text = re.sub(r'\b\d+\b', '', text)
    # Remove stop words
    tokens = text.split()
    tokens = [word for word in tokens if word.lower() not in ENGLISH_STOP_WORDS]
    return ' '.join(tokens)


data = fetch_20newsgroups(remove=('headers', 'footers', 'quotes'))

# Extract the data (text) and target (categories)
X = data.data  # The text documents
y = data.target  # The target categories

# Encode labels
label_encoder = LabelEncoder()
y_encoded_labels = label_encoder.fit_transform(y)

# Apply preprocessing to each document
X_preproc = [preprocessor(doc) for doc in X]

print(f"\nNumber of Words in BoW: {len(X_preproc)}")
print(f"\nBag of Words (BoW) / Vocabulary Keys:\n{X_preproc[:100]}")

# Train-test split
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y_encoded_labels, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)

max_vocab_size = 40000
max_sequence_length = 200

tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<UNK>")
tokenizer.fit_on_texts(X_train)

train_sequence = tokenizer.texts_to_sequences(X_train)
test_sequence = tokenizer.texts_to_sequences(X_test)
val_sequence = tokenizer.texts_to_sequences(X_val)

X_train = pad_sequences(train_sequence, max_sequence_length, padding='post')
X_test = pad_sequences(test_sequence, max_sequence_length, padding='post')
X_val = pad_sequences(val_sequence, max_sequence_length, padding='post')

# Number of classes
num_classes = np.max(y_encoded_labels) + 1

# One-hot encode the validation labels
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
y_val = to_categorical(y_val, num_classes)

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Define GRU Model
gru_model = Sequential([
    Embedding(input_dim=max_vocab_size, output_dim=128),
    GRU(128),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax'),
])

gru_model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

gpu_history = gru_model.fit(
    X_train, y_train,
    epochs=20, batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1,
    callbacks=[early_stopping]
)

val_loss, val_accuracy = gru_model.evaluate(X_val, y_val)
print(f"\nGRU Validation Accuracy: {val_accuracy:.4f}")

predictions = gru_model.predict(X_val)
predicted_class = np.argmax(predictions, axis=1)
true_labels = np.argmax(y_val, axis=1)

cm = confusion_matrix(true_labels, predicted_class)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot()
plt.title("Classification Matrix")
plt.show()
